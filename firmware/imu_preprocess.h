#pragma once

#include <stddef.h>

#include "app_config.h"

int imu_preprocess_init(void);
int imu_preprocess_window(
    const float input[IMU_EVENT_FRAMES * IMU_CHANNELS],
    float output[MODEL_IMU_FRAMES * MODEL_IMU_CHANNELS]);
int imu_preprocess_inplace(
    float values[MODEL_IMU_FRAMES * MODEL_IMU_CHANNELS]);
int imu_preprocess_finite(const float *values, size_t count);
