#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "app_config.h"

typedef struct
{
    /* Exact training order: ax, ay, az [m/s^2], gx, gy, gz [dps]. */
    float values[IMU_CHANNELS];
    uint32_t sequence;
    uint32_t timestamp_us;
    uint32_t timestamp_ms;
} imu_sample_t;

typedef struct
{
    uint8_t resolution_bits;
    uint8_t acc_odr;
    uint8_t acc_range;
    uint8_t acc_bwp;
    uint8_t acc_filter_perf;
    uint8_t gyr_odr;
    uint8_t gyr_range;
    uint8_t gyr_bwp;
    uint8_t gyr_noise_perf;
    uint8_t gyr_filter_perf;
    uint8_t power_ctrl;
    uint8_t power_conf;
    uint8_t internal_status;
    uint8_t startup_drdy_mask;
} sensor_imu_config_readback_t;

typedef struct
{
    uint32_t elapsed_us;
    uint32_t samples;
    float measured_hz;
    float mean_interval_us;
    uint32_t min_interval_us;
    uint32_t max_interval_us;
    uint32_t estimated_dropped_samples;
    uint32_t read_errors;
} sensor_imu_rate_stats_t;

int32_t sensor_imu_init(void);
bool sensor_imu_config_readback(sensor_imu_config_readback_t *readback);
uint32_t sensor_imu_next_sequence(void);
uint32_t sensor_imu_oldest_sequence(void);
bool sensor_imu_history_ready(void);
bool sensor_imu_read_sequence(uint32_t sequence, imu_sample_t *sample);
bool sensor_imu_snapshot_values(
    uint32_t first_sequence, size_t count, float *destination);
bool sensor_imu_rate_stats(sensor_imu_rate_stats_t *stats);
bool sensor_imu_faulted(void);
int32_t sensor_imu_fault_code(void);
