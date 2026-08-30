#pragma once

#include <stdint.h>

#include "app_config.h"

typedef struct
{
    uint32_t cycles;
    float latency_ms;
} model_timing_t;

int model_runner_init(void);
int model_runner_reset(void);
int model_runner_compute(
    const float imu[MODEL_IMU_FRAMES * MODEL_IMU_CHANNELS],
    const float audio[MODEL_AUDIO_FRAMES * MODEL_AUDIO_MELS],
    float embedding[MODEL_EMBEDDING_SIZE],
    model_timing_t *timing);
uint32_t model_runner_cycle_now(void);
float model_runner_cycles_to_ms(uint32_t cycles);
