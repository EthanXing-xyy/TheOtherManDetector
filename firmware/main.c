/******************************************************************************
 * TheOtherManDetector real-time Audio-IMU anomaly deployment.
 *
 * Author: Ethan_Xing
 * License: MIT
 *
 * Sensors are initialized once and remain continuously active.  Door state
 * only controls event latching, exact-window DSP, inference and re-arming.
 ******************************************************************************/

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "anomaly_detector.h"
#include "app_config.h"
#include "audio_preprocess.h"
#include "cy_retarget_io.h"
#include "cybsp.h"
#include "cyhal.h"
#include "door_state_machine.h"
#include "imu_preprocess.h"
#include "model_runner.h"
#include "microphone_capture.h"
#include "sensor_imu.h"
#include "timebase.h"
#include "validation_smoke.h"

#if !REAL_SENSOR_MODE && !ENABLE_VALIDATION_SMOKE_TEST
#error "Select REAL_SENSOR_MODE or ENABLE_VALIDATION_SMOKE_TEST"
#endif

typedef enum
{
    APP_STATE_INIT = 0,
    APP_STATE_WAIT_OPEN,
    APP_STATE_CAPTURE_OPEN_WINDOW,
    APP_STATE_PREPROCESS_AND_INFER,
    APP_STATE_WAIT_DOOR_STABLE,
    APP_STATE_WAIT_CLOSE,
    APP_STATE_CLOSE_COOLDOWN,
    APP_STATE_ERROR,
} app_state_t;

typedef struct
{
    bool active;
    bool imu_complete;
    bool audio_complete;
    uint32_t event_id;
    uint32_t t0_imu_sequence;
    uint32_t t0_us;
    uint64_t t0_audio_sample;
    uint64_t audio_first_sample;
    uint32_t next_audio_frame;
    int32_t error_code;
    const char *error_stage;
    uint64_t audio_dsp_cycle_sum;
    uint32_t audio_dsp_cycle_min;
    uint32_t audio_dsp_cycle_max;
    uint32_t imu_dsp_cycles;
    uint32_t capture_complete_us;
    uint32_t decision_us;
} event_context_t;

typedef struct
{
    uint64_t audio_samples_received;
    uint32_t imu_samples_received;
    uint32_t audio_overruns;
    uint32_t imu_dropped_samples;
    uint32_t open_trigger_count;
    uint32_t inference_count;
    uint32_t model_errors;
} app_counters_t;

/* Deliberately static: these symbols must remain visible in the linker map. */
static float model_imu_input[MODEL_IMU_FRAMES * MODEL_IMU_CHANNELS];
static float model_audio_input[MODEL_AUDIO_FRAMES * MODEL_AUDIO_MELS];
static float model_embedding[MODEL_EMBEDDING_SIZE];
static audio_preprocess_workspace_t audio_dsp_workspace;
static int16_t audio_frame_pcm[AUDIO_FRAME_SAMPLES];
static event_context_t current_event;
static app_counters_t app_counters;
static app_state_t app_state;
static uint32_t next_imu_sequence_to_service;

static const char *app_state_name(app_state_t state)
{
    switch (state)
    {
        case APP_STATE_INIT: return "INIT";
        case APP_STATE_WAIT_OPEN: return "WAIT_OPEN";
        case APP_STATE_CAPTURE_OPEN_WINDOW: return "CAPTURE_OPEN_WINDOW";
        case APP_STATE_PREPROCESS_AND_INFER: return "PREPROCESS_AND_INFER";
        case APP_STATE_WAIT_DOOR_STABLE: return "WAIT_DOOR_STABLE";
        case APP_STATE_WAIT_CLOSE: return "WAIT_CLOSE";
        case APP_STATE_CLOSE_COOLDOWN: return "CLOSE_COOLDOWN";
        case APP_STATE_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static void set_app_state(app_state_t state)
{
    if (app_state != state)
    {
        app_state = state;
        printf("[STATE] %s\r\n", app_state_name(state));
    }
}

static void safe_idle(void)
{
    app_state = APP_STATE_ERROR;
    fflush(stdout);
    for (;;)
    {
        /* Acquisition IRQs remain enabled so a debugger can inspect counters;
         * no result can be emitted from ERROR. */
        __WFI();
    }
}

static void init_failure(const char *module, int32_t code)
{
    printf("[INIT][%s] FAIL code=%ld\r\n", module, (long)code);
    safe_idle();
}

static bool audio_readback_valid(const microphone_capture_config_readback_t *readback)
{
    return (readback->clock_div_stage_1 == AUDIO_PDM_CLOCK_DIV_STAGE_1) &&
        (readback->mclk_div_stage_2 == AUDIO_PDM_MCLK_DIV_STAGE_2) &&
        (readback->output_div_stage_3 == AUDIO_PDM_OUTPUT_DIV_STAGE_3) &&
        (readback->sinc_rate == AUDIO_PDM_SINC_RATE) &&
        (readback->hardware_gain_enum == AUDIO_EXPECTED_HW_GAIN_ENUM);
}

static void print_sensor_rates(void)
{
    microphone_capture_rate_stats_t audio_rate;
    sensor_imu_rate_stats_t imu_rate;

    if (microphone_capture_rate_stats(&audio_rate))
    {
        printf(
            "[RATE][AUDIO] measured=%.3fHz samples=%lu elapsed_ms=%lu\r\n",
            audio_rate.measured_hz,
            (unsigned long)audio_rate.delta_samples,
            (unsigned long)audio_rate.elapsed_ms);
    }
    if (sensor_imu_rate_stats(&imu_rate))
    {
        printf(
            "[RATE][IMU] measured=%.3fHz mean_us=%.2f min_us=%lu max_us=%lu "
            "dropped=%lu errors=%lu\r\n",
            imu_rate.measured_hz,
            imu_rate.mean_interval_us,
            (unsigned long)imu_rate.min_interval_us,
            (unsigned long)imu_rate.max_interval_us,
            (unsigned long)imu_rate.estimated_dropped_samples,
            (unsigned long)imu_rate.read_errors);
    }
}

static void event_set_error(const char *stage, int32_t code)
{
    if (current_event.error_code == 0)
    {
        current_event.error_stage = stage;
        current_event.error_code = code;
    }
}

static void discard_current_event(void)
{
    printf(
        "[ERROR][%s] code=%ld event=%lu; result discarded\r\n",
        (current_event.error_stage != NULL) ? current_event.error_stage : "EVENT",
        (long)current_event.error_code,
        (unsigned long)current_event.event_id);
    if (event_detector_state() == DOOR_STATE_CAPTURE_OPEN_WINDOW)
    {
        (void)event_detector_begin_wait_open_settle();
    }
    memset(&current_event, 0, sizeof(current_event));
    set_app_state(APP_STATE_WAIT_DOOR_STABLE);
}

static void begin_event(const door_update_t *update)
{
    uint64_t t0_audio_sample;

    memset(&current_event, 0, sizeof(current_event));
    current_event.active = true;
    current_event.event_id = update->event_id;
    current_event.t0_imu_sequence = update->trigger_sequence;
    current_event.t0_us = update->trigger_us;
    current_event.audio_dsp_cycle_min = UINT32_MAX;
    ++app_counters.open_trigger_count;

    if (update->trigger_sequence < IMU_PRETRIGGER_FRAMES)
    {
        event_set_error("IMU_HISTORY", APP_ERROR_IMU_HISTORY);
    }
    else if (!microphone_capture_timestamp_to_sample(update->trigger_us, &t0_audio_sample))
    {
        event_set_error("AUDIO_TIMESTAMP", APP_ERROR_AUDIO_TIMESTAMP);
    }
    else if (t0_audio_sample < AUDIO_PRETRIGGER_SAMPLES)
    {
        event_set_error("AUDIO_HISTORY", APP_ERROR_AUDIO_TIMESTAMP);
    }
    else
    {
        current_event.t0_audio_sample = t0_audio_sample;
        current_event.audio_first_sample =
            t0_audio_sample - AUDIO_PRETRIGGER_SAMPLES;
    }

    printf("[EVENT] OPEN DETECTED\r\n");
    set_app_state(APP_STATE_CAPTURE_OPEN_WINDOW);
#if ENABLE_TRIGGER_DEBUG
    printf(
        "[TRIGGER] event=%lu t0_us=%lu imu_seq=%lu delta_axis=%.6f "
        "audio_t0=%lu\r\n",
        (unsigned long)current_event.event_id,
        (unsigned long)current_event.t0_us,
        (unsigned long)current_event.t0_imu_sequence,
        update->delta_ay,
        (unsigned long)current_event.t0_audio_sample);
#endif
}

static void service_audio_event_frames(void)
{
    if (!current_event.active || current_event.audio_complete ||
        (current_event.error_code != 0))
    {
        return;
    }

    while (current_event.next_audio_frame < AUDIO_FEATURE_FRAMES)
    {
        const uint64_t first = current_event.audio_first_sample +
            (uint64_t)current_event.next_audio_frame * AUDIO_HOP_SAMPLES;
        const uint64_t total = microphone_capture_total_samples();
        uint32_t start_cycles;
        uint32_t elapsed_cycles;
        int result;

        if (!microphone_capture_range_available(first, AUDIO_FRAME_SAMPLES))
        {
            if ((total > first) && ((total - first) > AUDIO_RING_SAMPLES))
            {
                ++app_counters.audio_overruns;
                event_set_error("AUDIO_RING", APP_ERROR_AUDIO_RING_OVERWRITE);
            }
            break;
        }
        if (!microphone_capture_copy_range(first, AUDIO_FRAME_SAMPLES, audio_frame_pcm))
        {
            event_set_error("AUDIO_COPY", APP_ERROR_AUDIO_RING_OVERWRITE);
            break;
        }

        start_cycles = model_runner_cycle_now();
        result = audio_preprocess_frame(
            audio_frame_pcm,
            &model_audio_input[current_event.next_audio_frame * AUDIO_MEL_BINS],
            &audio_dsp_workspace,
            NULL);
        elapsed_cycles = model_runner_cycle_now() - start_cycles;
        if (result != 0)
        {
            event_set_error("AUDIO_DSP", result);
            break;
        }
        current_event.audio_dsp_cycle_sum += elapsed_cycles;
        if (elapsed_cycles < current_event.audio_dsp_cycle_min)
        {
            current_event.audio_dsp_cycle_min = elapsed_cycles;
        }
        if (elapsed_cycles > current_event.audio_dsp_cycle_max)
        {
            current_event.audio_dsp_cycle_max = elapsed_cycles;
        }
        ++current_event.next_audio_frame;
    }
    current_event.audio_complete =
        current_event.next_audio_frame == AUDIO_FEATURE_FRAMES;
}

static void handle_door_update(const door_update_t *update)
{
    switch (update->kind)
    {
        case DOOR_UPDATE_OPEN_CONFIRMED:
            begin_event(update);
            break;
        case DOOR_UPDATE_IMU_WINDOW_COMPLETE:
            if (current_event.active)
            {
                current_event.imu_complete = true;
            }
            break;
        case DOOR_UPDATE_OPEN_SETTLED:
#if ENABLE_TRIGGER_DEBUG
            printf(
                "[TRIGGER] stable acc_var=%.9f gyro_var=%.9f\r\n",
                update->acc_var,
                update->gyro_var);
#endif
            set_app_state(APP_STATE_WAIT_CLOSE);
            break;
        case DOOR_UPDATE_CLOSE_DETECTED:
            printf("[EVENT] CLOSE DETECTED\r\n");
#if ENABLE_TRIGGER_DEBUG
            printf("[TRIGGER] close_delta=%.6f\r\n", update->delta_ay);
#endif
            set_app_state(APP_STATE_CLOSE_COOLDOWN);
            break;
        case DOOR_UPDATE_REARMED:
            if (model_runner_reset() != 0)
            {
                printf("[ERROR][MODEL_RESET] code=%d\r\n", APP_ERROR_MODEL_RESET);
                safe_idle();
            }
            set_app_state(APP_STATE_WAIT_OPEN);
            break;
        case DOOR_UPDATE_NONE:
        default:
            break;
    }
}

/* Returns early at exact IMU window completion so PREPROCESS_AND_INFER can
 * transition the detector before any queued post-window samples are consumed. */
static void service_imu_samples(void)
{
    uint32_t oldest = sensor_imu_oldest_sequence();
    uint32_t next = sensor_imu_next_sequence();

    if (next_imu_sequence_to_service < oldest)
    {
        app_counters.imu_dropped_samples +=
            oldest - next_imu_sequence_to_service;
        printf(
            "[ERROR][IMU_RING] code=%d dropped=%lu\r\n",
            APP_ERROR_IMU_HISTORY,
            (unsigned long)(oldest - next_imu_sequence_to_service));
        safe_idle();
    }

    while (next_imu_sequence_to_service < next)
    {
        imu_sample_t sample;
        door_update_t update;
        if (!sensor_imu_read_sequence(next_imu_sequence_to_service, &sample))
        {
            printf("[ERROR][IMU_READ] code=%d\r\n", APP_ERROR_IMU_HISTORY);
            safe_idle();
        }
        ++next_imu_sequence_to_service;
        app_counters.imu_samples_received = next_imu_sequence_to_service;
        update = event_detector_accept(
            &sample,
            microphone_capture_history_ready() && sensor_imu_history_ready());
        handle_door_update(&update);
        if (update.kind == DOOR_UPDATE_IMU_WINDOW_COMPLETE)
        {
            return;
        }
    }
}

static void print_result(const anomaly_result_t *result)
{
    printf("[RESULT]\r\n");
    printf("Score     : %.6f\r\n", result->score);
    printf("Threshold : %.6f\r\n", result->threshold);
    printf("Decision  : %s\r\n", result->anomaly ? "ANOMALY" : "NORMAL");
}

static void process_complete_event(void)
{
    model_timing_t model_timing = {0};
    anomaly_result_t anomaly = {0};
    uint32_t imu_dsp_start;
    int result;

    if (!current_event.active || !current_event.imu_complete ||
        !current_event.audio_complete)
    {
        return;
    }
    current_event.capture_complete_us = timebase_now_us();
    printf("[CAPTURE] COMPLETE\r\n");
#if ENABLE_SAMPLE_WINDOW_DEBUG
    printf(
        "[WINDOW] audio=%u imu=%u t0_frame=62 t0_offset=320 "
        "audio_t0=%lu imu_t0=%lu\r\n",
        (unsigned int)AUDIO_EVENT_SAMPLES,
        (unsigned int)IMU_EVENT_FRAMES,
        (unsigned long)current_event.t0_audio_sample,
        (unsigned long)current_event.t0_imu_sequence);
#endif

    if (current_event.error_code != 0)
    {
        discard_current_event();
        return;
    }
    if (!event_detector_begin_wait_open_settle())
    {
        event_set_error("STATE", APP_ERROR_WINDOW_COUNT);
        discard_current_event();
        return;
    }
    set_app_state(APP_STATE_PREPROCESS_AND_INFER);

    if (!sensor_imu_snapshot_values(
            current_event.t0_imu_sequence - IMU_PRETRIGGER_FRAMES,
            IMU_EVENT_FRAMES,
            model_imu_input))
    {
        event_set_error("IMU_SNAPSHOT", APP_ERROR_IMU_HISTORY);
        discard_current_event();
        return;
    }
    imu_dsp_start = model_runner_cycle_now();
    result = imu_preprocess_inplace(model_imu_input);
    current_event.imu_dsp_cycles = model_runner_cycle_now() - imu_dsp_start;
    if (result != 0)
    {
        event_set_error("IMU_DSP", result);
        discard_current_event();
        return;
    }
    if (!audio_preprocess_finite(
            model_audio_input, MODEL_AUDIO_FRAMES * MODEL_AUDIO_MELS) ||
        !imu_preprocess_finite(
            model_imu_input, MODEL_IMU_FRAMES * MODEL_IMU_CHANNELS))
    {
        event_set_error("MODEL_INPUT", APP_ERROR_NONFINITE);
        discard_current_event();
        return;
    }

    result = model_runner_compute(
        model_imu_input,
        model_audio_input,
        model_embedding,
        &model_timing);
    if (result != 0)
    {
        ++app_counters.model_errors;
        event_set_error("MODEL", result);
        discard_current_event();
        return;
    }
    result = anomaly_detector_run(model_embedding, &anomaly);
    if (result != 0)
    {
        event_set_error("K2", result);
        discard_current_event();
        return;
    }
    ++app_counters.inference_count;
    print_result(&anomaly);
    current_event.decision_us = timebase_now_us();

#if ENABLE_DSP_DEBUG
    printf(
        "[TIMING] audio_frame_ms min=%.3f mean=%.3f max=%.3f "
        "audio_180_ms=%.3f imu_ms=%.3f TMAI_ms=%.3f K2_ms=%.3f "
        "end_to_end_ms=%.3f\r\n",
        model_runner_cycles_to_ms(current_event.audio_dsp_cycle_min),
        model_runner_cycles_to_ms(
            (uint32_t)(current_event.audio_dsp_cycle_sum / AUDIO_FEATURE_FRAMES)),
        model_runner_cycles_to_ms(current_event.audio_dsp_cycle_max),
        model_runner_cycles_to_ms((uint32_t)current_event.audio_dsp_cycle_sum),
        model_runner_cycles_to_ms(current_event.imu_dsp_cycles),
        model_timing.latency_ms,
        anomaly.latency_ms,
        (float)(current_event.decision_us - current_event.capture_complete_us) / 1000.0f);
#endif

    memset(&current_event, 0, sizeof(current_event));
    set_app_state(APP_STATE_WAIT_DOOR_STABLE);
}

static void wait_for_pretrigger_history(void)
{
    const uint32_t start_ms = timebase_now_ms();
    while (!microphone_capture_history_ready() || !sensor_imu_history_ready())
    {
        if (microphone_capture_faulted())
        {
            printf(
                "[ERROR][AUDIO_STREAM] code=%ld\r\n",
                (long)microphone_capture_fault_code());
            safe_idle();
        }
        if (sensor_imu_faulted())
        {
            printf(
                "[ERROR][IMU_STREAM] code=%ld\r\n",
                (long)sensor_imu_fault_code());
            safe_idle();
        }
        if (timebase_elapsed_ms(start_ms, timebase_now_ms()) > BOOT_SENSOR_TIMEOUT_MS)
        {
            printf("[ERROR][SENSOR_HISTORY] code=%d\r\n", APP_ERROR_WINDOW_COUNT);
            safe_idle();
        }
        __WFI();
    }
}

int main(void)
{
    cy_rslt_t hal_result;
    int32_t result;

    app_state = APP_STATE_INIT;
    memset(&app_counters, 0, sizeof(app_counters));
    hal_result = cybsp_init();
    if (hal_result != CY_RSLT_SUCCESS)
    {
        for (;;) { __WFI(); }
    }
    __enable_irq();
    hal_result = cy_retarget_io_init_fc(
        CYBSP_DEBUG_UART_TX,
        CYBSP_DEBUG_UART_RX,
        CYBSP_DEBUG_UART_CTS,
        CYBSP_DEBUG_UART_RTS,
        CY_RETARGET_IO_BAUDRATE);
    if (hal_result != CY_RSLT_SUCCESS)
    {
        for (;;) { __WFI(); }
    }

    printf("\r\n[BOOT] TheOtherManDetector\r\n");
    printf("[INIT][BSP] OK\r\n");
    printf("[INIT][UART] OK\r\n");
    result = timebase_init();
    if (result != 0)
    {
        init_failure("TIMEBASE", result);
    }

#if ENABLE_VALIDATION_SMOKE_TEST
    result = audio_preprocess_init();
    if (result != 0) init_failure("DSP", result);
    printf("[INIT][DSP] OK\r\n");
    result = imu_preprocess_init();
    if (result != 0) init_failure("DSP", result);
    result = model_runner_init();
    if (result != 0) init_failure("MODEL", result);
    printf("[INIT][MODEL] OK\r\n");
    result = anomaly_detector_init();
    if (result != 0) init_failure("K2", result);
    printf("[INIT][K2] OK\r\n");
    result = validation_smoke_run();
    if (result != 0) safe_idle();
    for (;;) { __WFI(); }
#else
    result = sensor_imu_init();
    if (result != 0)
    {
        init_failure("IMU", result);
    }
    sensor_imu_config_readback_t imu_config;
    if (!sensor_imu_config_readback(&imu_config))
    {
        init_failure("IMU", APP_ERROR_IMU_CONFIG);
    }
    printf("[INIT][IMU] OK\r\n");
    printf("[IMU] ODR=100Hz\r\n");
    printf("[IMU] ACC_RANGE=8g\r\n");
    printf("[IMU] GYRO_RANGE=500dps\r\n");
    printf("[IMU] MODE=COMBINED order=Ax,Ay,Az,Gx,Gy,Gz units=m/s2,dps\r\n");
    printf(
        "[IMU] READBACK acc_odr=0x%02x acc_range=0x%02x "
        "gyr_odr=0x%02x gyr_range=0x%02x power=0x%02x status=0x%02x\r\n",
        imu_config.acc_odr,
        imu_config.acc_range,
        imu_config.gyr_odr,
        imu_config.gyr_range,
        imu_config.power_ctrl,
        imu_config.internal_status);

    result = microphone_capture_init();
    if (result == 0)
    {
        result = microphone_capture_start();
    }
    if (result != 0)
    {
        init_failure("AUDIO", result);
    }
    microphone_capture_config_readback_t audio_config;
    if (!microphone_capture_config_readback(&audio_config) ||
        !audio_readback_valid(&audio_config))
    {
        init_failure("AUDIO", APP_ERROR_AUDIO_CONFIG);
    }
    printf("[INIT][AUDIO] OK\r\n");
    printf("[AUDIO] RATE=16000Hz\r\n");
    printf("[AUDIO] CHANNELS=1 mode=MONO_LEFT PCM=S16\r\n");
    printf(
        "[AUDIO] GAIN=5 HAL_REQUEST=10x0.5dB HW_ENUM=%lu HW_GAIN=%.1fdB\r\n",
        (unsigned long)audio_config.hardware_gain_enum,
        audio_config.hardware_gain_db);
    printf("[AUDIO] CONVERT=ON behavior=PCM16/32768.0f\r\n");

    result = audio_preprocess_init();
    if (result == 0)
    {
        result = imu_preprocess_init();
    }
    if (result != 0)
    {
        init_failure("DSP", result);
    }
    printf("[INIT][DSP] OK\r\n");
    result = model_runner_init();
    if (result != 0)
    {
        init_failure("MODEL", result);
    }
    printf("[INIT][MODEL] OK\r\n");
    result = anomaly_detector_init();
    if (result != 0)
    {
        init_failure("K2", result);
    }
    printf("[INIT][K2] OK\r\n");

    event_detector_init();
    wait_for_pretrigger_history();
    next_imu_sequence_to_service = sensor_imu_oldest_sequence();
    app_counters.audio_samples_received = microphone_capture_total_samples();
    print_sensor_rates();
    app_state = APP_STATE_WAIT_OPEN;
    printf("[READY] WAIT_OPEN\r\n");

    for (;;)
    {
        if (microphone_capture_faulted())
        {
            printf(
                "[ERROR][AUDIO_STREAM] code=%ld\r\n",
                (long)microphone_capture_fault_code());
            safe_idle();
        }
        if (sensor_imu_faulted())
        {
            printf(
                "[ERROR][IMU_STREAM] code=%ld\r\n",
                (long)sensor_imu_fault_code());
            safe_idle();
        }

        app_counters.audio_samples_received = microphone_capture_total_samples();
        service_audio_event_frames();
        if (current_event.active && current_event.imu_complete &&
            current_event.audio_complete)
        {
            process_complete_event();
            continue;
        }
        if (current_event.active && (current_event.error_code != 0))
        {
            discard_current_event();
            continue;
        }

        service_imu_samples();
        service_audio_event_frames();
        if (current_event.active && current_event.imu_complete &&
            current_event.audio_complete)
        {
            process_complete_event();
            continue;
        }
        if (current_event.active && (current_event.error_code != 0))
        {
            discard_current_event();
            continue;
        }
        __WFI();
    }
#endif
}
