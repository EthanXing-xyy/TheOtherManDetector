#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "app_config.h"

typedef enum
{
    AUDIO_FAULT_NONE = 0,
    AUDIO_FAULT_RX_OVERFLOW,
    AUDIO_FAULT_ASYNC_REARM,
    AUDIO_FAULT_HAL,
    AUDIO_FAULT_DMA,
} microphone_capture_fault_kind_t;

typedef struct
{
    microphone_capture_fault_kind_t kind;
    int32_t code;
} microphone_capture_fault_info_t;

typedef struct
{
    uint32_t pll_hz;
    uint32_t audio_clock_hz;
    uint32_t pdm_clock_ctl;
    uint32_t clock_div_stage_1;
    uint32_t mclk_div_stage_2;
    uint32_t output_div_stage_3;
    uint32_t sinc_rate;
    uint32_t hardware_gain_enum;
    float hardware_gain_db;
} microphone_capture_config_readback_t;

typedef struct
{
    uint32_t elapsed_ms;
    uint64_t delta_samples;
    float measured_hz;
    float error_percent;
} microphone_capture_rate_stats_t;

int32_t microphone_capture_init(void);
int32_t microphone_capture_start(void);
bool microphone_capture_config_readback(microphone_capture_config_readback_t *readback);
bool microphone_capture_rate_stats(microphone_capture_rate_stats_t *stats);
uint64_t microphone_capture_total_samples(void);
uint32_t microphone_capture_completed_blocks(void);
bool microphone_capture_history_ready(void);
bool microphone_capture_timestamp_to_sample(uint32_t timestamp_us, uint64_t *sample_index);
bool microphone_capture_target_available(uint64_t t0_sample_index);
bool microphone_capture_range_available(uint64_t first_sample, size_t count);
bool microphone_capture_copy_range(
    uint64_t first_sample, size_t count, int16_t *destination);
bool microphone_capture_snapshot_event(uint64_t t0_sample_index, int16_t *destination);
bool microphone_capture_faulted(void);
int32_t microphone_capture_fault_code(void);
microphone_capture_fault_info_t microphone_capture_fault_info(void);
const char *microphone_capture_fault_name(microphone_capture_fault_kind_t kind);

