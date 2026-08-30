#include "imu_preprocess.h"

#include <math.h>

#include "imu_preprocess_constants.h"

int imu_preprocess_init(void)
{
    for (size_t axis = 0U; axis < IMU_CHANNELS; ++axis)
    {
        if (!(imu_normal_train_std_6[axis] > 0.0f))
        {
            return APP_ERROR_PREPROCESS;
        }
    }
    return 0;
}

int imu_preprocess_window(
    const float input[IMU_EVENT_FRAMES * IMU_CHANNELS],
    float output[MODEL_IMU_FRAMES * MODEL_IMU_CHANNELS])
{
    if ((input == NULL) || (output == NULL))
    {
        return APP_ERROR_PREPROCESS;
    }
    for (size_t frame = 0U; frame < IMU_EVENT_FRAMES; ++frame)
    {
        for (size_t axis = 0U; axis < IMU_CHANNELS; ++axis)
        {
            const size_t index = frame * IMU_CHANNELS + axis;
            output[index] = (input[index] - imu_normal_train_mean_6[axis]) /
                imu_normal_train_std_6[axis];
        }
    }
    return imu_preprocess_finite(
        output, MODEL_IMU_FRAMES * MODEL_IMU_CHANNELS) ?
        0 : APP_ERROR_NONFINITE;
}

int imu_preprocess_inplace(
    float values[MODEL_IMU_FRAMES * MODEL_IMU_CHANNELS])
{
    return imu_preprocess_window(values, values);
}

int imu_preprocess_finite(const float *values, size_t count)
{
    if (values == NULL)
    {
        return 0;
    }
    for (size_t index = 0U; index < count; ++index)
    {
        if (!isfinite(values[index]))
        {
            return 0;
        }
    }
    return 1;
}
