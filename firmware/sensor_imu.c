#include "sensor_imu.h"

#include <limits.h>
#include <string.h>

#include "bmi270.h"
#include "cybsp.h"
#include "cyhal.h"
#include "timebase.h"

#define BMI270_ADDRESS (BMI2_I2C_PRIM_ADDR)
#define SENSOR_COUNT (2U)
#define BMI270_COMBINED_DATA_ADDR (0x0CU)
#define BMI270_COMBINED_DATA_BYTES (12U)
#define I2C_TIMEOUT_MS (10U)
#define READ_WRITE_LEN (46U)
#define GRAVITY_EARTH_MPS2 (9.80665f)
#define IMU_RING_MASK (IMU_HISTORY_FRAMES - 1U)
#define IMU_STARTUP_TIMEOUT_US (500000U)

static cyhal_i2c_t imu_i2c;
static cyhal_timer_t imu_timer;
static struct bmi2_dev imu_device;
static imu_sample_t imu_history[IMU_HISTORY_FRAMES];
static volatile uint32_t imu_next_sequence_value;
static volatile bool imu_fault;
static volatile int32_t imu_error;
static sensor_imu_config_readback_t imu_readback;
static bool imu_readback_valid;

static volatile uint32_t stats_first_timestamp_us;
static volatile uint32_t stats_last_timestamp_us;
static volatile uint64_t stats_interval_sum_us;
static volatile uint32_t stats_interval_count;
static volatile uint32_t stats_min_interval_us;
static volatile uint32_t stats_max_interval_us;
static volatile uint32_t stats_estimated_drops;
static volatile uint32_t stats_read_errors;

static BMI2_INTF_RETURN_TYPE bmi_i2c_read(
    uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr)
{
    cy_rslt_t result = cyhal_i2c_master_mem_read(
        (cyhal_i2c_t *)intf_ptr,
        BMI270_ADDRESS,
        reg_addr,
        1U,
        reg_data,
        (uint16_t)len,
        I2C_TIMEOUT_MS);
    return (result == CY_RSLT_SUCCESS) ?
        BMI2_INTF_RET_SUCCESS : BMI2_E_COM_FAIL;
}

static BMI2_INTF_RETURN_TYPE bmi_i2c_write(
    uint8_t reg_addr, const uint8_t *reg_data, uint32_t len, void *intf_ptr)
{
    cy_rslt_t result = cyhal_i2c_master_mem_write(
        (cyhal_i2c_t *)intf_ptr,
        BMI270_ADDRESS,
        reg_addr,
        1U,
        reg_data,
        (uint16_t)len,
        I2C_TIMEOUT_MS);
    return (result == CY_RSLT_SUCCESS) ?
        BMI2_INTF_RET_SUCCESS : BMI2_E_COM_FAIL;
}

static void bmi_delay_us(uint32_t us, void *intf_ptr)
{
    (void)intf_ptr;
    cyhal_system_delay_us(us);
}

static int16_t load_i16_le(const uint8_t *bytes)
{
    return (int16_t)(((uint16_t)bytes[1] << 8U) | (uint16_t)bytes[0]);
}

static void set_fault_once(int32_t code)
{
    if (!imu_fault)
    {
        imu_error = code;
        imu_fault = true;
    }
}

/* This ISR is the continuous acquisition owner.  It performs one bounded
 * 12-byte combined register burst every 10 ms.  Therefore TMAI_compute in the
 * foreground cannot pause or reinitialize the IMU stream. */
static void imu_timer_callback(void *callback_arg, cyhal_timer_event_t event)
{
    uint8_t bytes[BMI270_COMBINED_DATA_BYTES];
    imu_sample_t sample;
    uint32_t timestamp_us;
    uint32_t sequence;
    cy_rslt_t result;

    (void)callback_arg;
    (void)event;
    if (imu_fault)
    {
        return;
    }

    timestamp_us = timebase_now_us();
    result = cyhal_i2c_master_mem_read(
        &imu_i2c,
        BMI270_ADDRESS,
        BMI270_COMBINED_DATA_ADDR,
        1U,
        bytes,
        BMI270_COMBINED_DATA_BYTES,
        I2C_TIMEOUT_MS);
    if (result != CY_RSLT_SUCCESS)
    {
        ++stats_read_errors;
        set_fault_once((int32_t)result);
        return;
    }

    const int16_t ax = load_i16_le(&bytes[0]);
    const int16_t ay = load_i16_le(&bytes[2]);
    const int16_t az = load_i16_le(&bytes[4]);
    const int16_t gx = load_i16_le(&bytes[6]);
    const int16_t gy = load_i16_le(&bytes[8]);
    const int16_t gz = load_i16_le(&bytes[10]);
    const float accel_scale =
        (GRAVITY_EARTH_MPS2 * (float)IMU_ACCEL_RANGE_G) / 32768.0f;
    const float gyro_scale = (float)IMU_GYRO_RANGE_DPS / 32768.0f;

    sample.values[0] = (float)ax * accel_scale;
    sample.values[1] = (float)ay * accel_scale;
    sample.values[2] = (float)az * accel_scale;
    sample.values[3] = (float)gx * gyro_scale;
    sample.values[4] = (float)gy * gyro_scale;
    sample.values[5] = (float)gz * gyro_scale;
    sequence = imu_next_sequence_value;
    sample.sequence = sequence;
    sample.timestamp_us = timestamp_us;
    sample.timestamp_ms = timestamp_us / 1000U;

    if (sequence == 0U)
    {
        stats_first_timestamp_us = timestamp_us;
        stats_min_interval_us = UINT32_MAX;
    }
    else
    {
        const uint32_t interval = timestamp_us - stats_last_timestamp_us;
        stats_interval_sum_us += interval;
        ++stats_interval_count;
        if (interval < stats_min_interval_us)
        {
            stats_min_interval_us = interval;
        }
        if (interval > stats_max_interval_us)
        {
            stats_max_interval_us = interval;
        }
        if (interval > (IMU_SAMPLE_PERIOD_US + IMU_SAMPLE_PERIOD_US / 2U))
        {
            uint32_t periods =
                (interval + IMU_SAMPLE_PERIOD_US / 2U) / IMU_SAMPLE_PERIOD_US;
            if (periods > 1U)
            {
                stats_estimated_drops += periods - 1U;
            }
        }
    }
    stats_last_timestamp_us = timestamp_us;

    imu_history[sequence & IMU_RING_MASK] = sample;
    __DMB();
    imu_next_sequence_value = sequence + 1U;
}

static int32_t synchronize_stream(void)
{
    const uint8_t required = BMI2_DRDY_ACC | BMI2_DRDY_GYR;
    uint8_t seen = 0U;
    uint32_t start_us = timebase_now_us();

    while ((seen & required) != required)
    {
        struct bmi2_sens_data data = {0};
        int8_t result = bmi2_get_sensor_data(&data, &imu_device);
        if (result != BMI2_OK)
        {
            return (int32_t)result;
        }
        seen |= data.status & required;
        if ((timebase_now_us() - start_us) >= IMU_STARTUP_TIMEOUT_US)
        {
            return APP_ERROR_IMU_SENSOR;
        }
        if ((seen & required) != required)
        {
            cyhal_system_delay_us(1000U);
        }
    }
    imu_readback.startup_drdy_mask = seen;
    return 0;
}

static int32_t configure_bmi270(void)
{
    struct bmi2_sens_config config[SENSOR_COUNT] = {0};
    uint8_t sensors[SENSOR_COUNT] = { BMI2_ACCEL, BMI2_GYRO };
    int8_t result;

    memset(&imu_device, 0, sizeof(imu_device));
    imu_device.intf = BMI2_I2C_INTF;
    imu_device.read = bmi_i2c_read;
    imu_device.write = bmi_i2c_write;
    imu_device.delay_us = bmi_delay_us;
    imu_device.intf_ptr = &imu_i2c;
    imu_device.read_write_len = READ_WRITE_LEN;
    imu_device.config_file_ptr = NULL;

    result = bmi270_init(&imu_device);
    if (result != BMI2_OK)
    {
        return (int32_t)result;
    }
    if (imu_device.resolution != 16U)
    {
        return APP_ERROR_IMU_RESOLUTION;
    }

    config[0].type = BMI2_ACCEL;
    config[1].type = BMI2_GYRO;
    result = bmi2_get_sensor_config(config, SENSOR_COUNT, &imu_device);
    if (result != BMI2_OK)
    {
        return (int32_t)result;
    }
    config[0].cfg.acc.odr = BMI2_ACC_ODR_100HZ;
    config[0].cfg.acc.range = BMI2_ACC_RANGE_8G;
    config[0].cfg.acc.bwp = BMI2_ACC_OSR4_AVG1;
    config[0].cfg.acc.filter_perf = BMI2_PERF_OPT_MODE;
    config[1].cfg.gyr.odr = BMI2_GYR_ODR_100HZ;
    config[1].cfg.gyr.range = BMI2_GYR_RANGE_500;
    config[1].cfg.gyr.bwp = BMI2_GYR_NORMAL_MODE;
    config[1].cfg.gyr.noise_perf = BMI2_POWER_OPT_MODE;
    config[1].cfg.gyr.filter_perf = BMI2_PERF_OPT_MODE;
    result = bmi2_set_sensor_config(config, SENSOR_COUNT, &imu_device);
    if (result != BMI2_OK)
    {
        return (int32_t)result;
    }

    memset(config, 0, sizeof(config));
    config[0].type = BMI2_ACCEL;
    config[1].type = BMI2_GYRO;
    result = bmi2_get_sensor_config(config, SENSOR_COUNT, &imu_device);
    if (result != BMI2_OK)
    {
        return (int32_t)result;
    }
    imu_readback.resolution_bits = imu_device.resolution;
    imu_readback.acc_odr = config[0].cfg.acc.odr;
    imu_readback.acc_range = config[0].cfg.acc.range;
    imu_readback.acc_bwp = config[0].cfg.acc.bwp;
    imu_readback.acc_filter_perf = config[0].cfg.acc.filter_perf;
    imu_readback.gyr_odr = config[1].cfg.gyr.odr;
    imu_readback.gyr_range = config[1].cfg.gyr.range;
    imu_readback.gyr_bwp = config[1].cfg.gyr.bwp;
    imu_readback.gyr_noise_perf = config[1].cfg.gyr.noise_perf;
    imu_readback.gyr_filter_perf = config[1].cfg.gyr.filter_perf;
    if ((imu_readback.acc_odr != BMI2_ACC_ODR_100HZ) ||
        (imu_readback.acc_range != BMI2_ACC_RANGE_8G) ||
        (imu_readback.gyr_odr != BMI2_GYR_ODR_100HZ) ||
        (imu_readback.gyr_range != BMI2_GYR_RANGE_500))
    {
        return APP_ERROR_IMU_CONFIG;
    }

    result = bmi2_sensor_enable(sensors, SENSOR_COUNT, &imu_device);
    if (result != BMI2_OK)
    {
        return (int32_t)result;
    }
    result = bmi2_get_regs(BMI2_PWR_CTRL_ADDR, &imu_readback.power_ctrl, 1U, &imu_device);
    if (result != BMI2_OK)
    {
        return (int32_t)result;
    }
    result = bmi2_get_regs(BMI2_PWR_CONF_ADDR, &imu_readback.power_conf, 1U, &imu_device);
    if (result != BMI2_OK)
    {
        return (int32_t)result;
    }
    result = bmi2_get_internal_status(&imu_readback.internal_status, &imu_device);
    if (result != BMI2_OK)
    {
        return (int32_t)result;
    }
    if (((imu_readback.power_ctrl & (BMI2_ACC_EN_MASK | BMI2_GYR_EN_MASK)) !=
         (BMI2_ACC_EN_MASK | BMI2_GYR_EN_MASK)) ||
        (imu_readback.internal_status != BMI2_INIT_OK))
    {
        return APP_ERROR_IMU_CONFIG;
    }
    {
        int32_t sync_result = synchronize_stream();
        if (sync_result != 0)
        {
            return sync_result;
        }
    }
    imu_readback_valid = true;
    return 0;
}

static int32_t configure_timer(void)
{
    const cyhal_timer_cfg_t config = {
        .is_continuous = true,
        .direction = CYHAL_TIMER_DIR_UP,
        .is_compare = false,
        .period = IMU_SAMPLE_PERIOD_US - 1U,
        .compare_value = 0U,
        .value = 0U,
    };
    cy_rslt_t result = cyhal_timer_init(&imu_timer, NC, NULL);
    if (result != CY_RSLT_SUCCESS)
    {
        return (int32_t)result;
    }
    result = cyhal_timer_set_frequency(&imu_timer, 1000000U);
    if (result != CY_RSLT_SUCCESS)
    {
        return (int32_t)result;
    }
    result = cyhal_timer_configure(&imu_timer, &config);
    if (result != CY_RSLT_SUCCESS)
    {
        return (int32_t)result;
    }
    cyhal_timer_register_callback(&imu_timer, imu_timer_callback, NULL);
    cyhal_timer_enable_event(
        &imu_timer, CYHAL_TIMER_IRQ_TERMINAL_COUNT, IMU_ISR_PRIORITY, true);
    return (int32_t)cyhal_timer_start(&imu_timer);
}

int32_t sensor_imu_init(void)
{
    const cyhal_i2c_cfg_t i2c_config = {
        .is_slave = false,
        .address = 0U,
        .frequencyhal_hz = IMU_I2C_FREQUENCY_HZ,
    };
    cy_rslt_t hal_result;
    int32_t result;

    memset(imu_history, 0, sizeof(imu_history));
    memset(&imu_readback, 0, sizeof(imu_readback));
    imu_next_sequence_value = 0U;
    imu_fault = false;
    imu_error = 0;
    imu_readback_valid = false;
    stats_first_timestamp_us = 0U;
    stats_last_timestamp_us = 0U;
    stats_interval_sum_us = 0U;
    stats_interval_count = 0U;
    stats_min_interval_us = UINT32_MAX;
    stats_max_interval_us = 0U;
    stats_estimated_drops = 0U;
    stats_read_errors = 0U;

    hal_result = cyhal_i2c_init(&imu_i2c, CYBSP_I2C_SDA, CYBSP_I2C_SCL, NULL);
    if (hal_result != CY_RSLT_SUCCESS)
    {
        return (int32_t)hal_result;
    }
    hal_result = cyhal_i2c_configure(&imu_i2c, &i2c_config);
    if (hal_result != CY_RSLT_SUCCESS)
    {
        return (int32_t)hal_result;
    }
    result = configure_bmi270();
    if (result == 0)
    {
        result = configure_timer();
    }
    if (result != 0)
    {
        set_fault_once(result);
    }
    return result;
}

bool sensor_imu_config_readback(sensor_imu_config_readback_t *readback)
{
    if ((readback == NULL) || !imu_readback_valid)
    {
        return false;
    }
    *readback = imu_readback;
    return true;
}

uint32_t sensor_imu_next_sequence(void)
{
    return imu_next_sequence_value;
}

uint32_t sensor_imu_oldest_sequence(void)
{
    uint32_t next = sensor_imu_next_sequence();
    return (next > IMU_HISTORY_FRAMES) ? next - IMU_HISTORY_FRAMES : 0U;
}

bool sensor_imu_history_ready(void)
{
    return !sensor_imu_faulted() &&
        (sensor_imu_next_sequence() >= IMU_PRETRIGGER_FRAMES);
}

bool sensor_imu_read_sequence(uint32_t sequence, imu_sample_t *sample)
{
    uint32_t saved_interrupt_state;
    uint32_t next;
    uint32_t oldest;

    if ((sample == NULL) || sensor_imu_faulted())
    {
        return false;
    }
    saved_interrupt_state = cyhal_system_critical_section_enter();
    next = imu_next_sequence_value;
    oldest = (next > IMU_HISTORY_FRAMES) ? next - IMU_HISTORY_FRAMES : 0U;
    if ((sequence < oldest) || (sequence >= next) ||
        (imu_history[sequence & IMU_RING_MASK].sequence != sequence))
    {
        cyhal_system_critical_section_exit(saved_interrupt_state);
        return false;
    }
    *sample = imu_history[sequence & IMU_RING_MASK];
    cyhal_system_critical_section_exit(saved_interrupt_state);
    return true;
}

bool sensor_imu_snapshot_values(
    uint32_t first_sequence, size_t count, float *destination)
{
    uint32_t saved_interrupt_state;
    uint32_t next;
    uint32_t oldest;

    if ((destination == NULL) || (count > IMU_HISTORY_FRAMES) ||
        sensor_imu_faulted())
    {
        return false;
    }
    saved_interrupt_state = cyhal_system_critical_section_enter();
    next = imu_next_sequence_value;
    oldest = (next > IMU_HISTORY_FRAMES) ? next - IMU_HISTORY_FRAMES : 0U;
    if ((first_sequence < oldest) ||
        ((uint64_t)first_sequence + count > next))
    {
        cyhal_system_critical_section_exit(saved_interrupt_state);
        return false;
    }
    for (size_t frame = 0U; frame < count; ++frame)
    {
        uint32_t sequence = first_sequence + (uint32_t)frame;
        const imu_sample_t *sample = &imu_history[sequence & IMU_RING_MASK];
        if (sample->sequence != sequence)
        {
            cyhal_system_critical_section_exit(saved_interrupt_state);
            return false;
        }
        memcpy(
            destination + frame * IMU_CHANNELS,
            sample->values,
            IMU_CHANNELS * sizeof(float));
    }
    cyhal_system_critical_section_exit(saved_interrupt_state);
    return !sensor_imu_faulted();
}

bool sensor_imu_rate_stats(sensor_imu_rate_stats_t *stats)
{
    uint32_t saved_interrupt_state;
    uint32_t next;
    uint32_t interval_count;
    uint64_t interval_sum;
    uint32_t first;
    uint32_t last;

    if (stats == NULL)
    {
        return false;
    }
    saved_interrupt_state = cyhal_system_critical_section_enter();
    next = imu_next_sequence_value;
    interval_count = stats_interval_count;
    interval_sum = stats_interval_sum_us;
    first = stats_first_timestamp_us;
    last = stats_last_timestamp_us;
    stats->min_interval_us = stats_min_interval_us;
    stats->max_interval_us = stats_max_interval_us;
    stats->estimated_dropped_samples = stats_estimated_drops;
    stats->read_errors = stats_read_errors;
    cyhal_system_critical_section_exit(saved_interrupt_state);

    stats->samples = next;
    stats->elapsed_us = last - first;
    stats->mean_interval_us = (interval_count != 0U) ?
        (float)interval_sum / (float)interval_count : 0.0f;
    stats->measured_hz = (stats->elapsed_us != 0U) ?
        ((float)(next - 1U) * 1000000.0f) / (float)stats->elapsed_us : 0.0f;
    return next >= 2U;
}

bool sensor_imu_faulted(void)
{
    return imu_fault;
}

int32_t sensor_imu_fault_code(void)
{
    return imu_error;
}
