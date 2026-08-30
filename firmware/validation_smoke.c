#include "validation_smoke.h"

#include "app_config.h"

#if ENABLE_VALIDATION_SMOKE_TEST

#include <math.h>
#include <stdio.h>

#include "anomaly_detector.h"
#include "model_runner.h"
#include "model/validation_sample.h"

int validation_smoke_run(void)
{
    float output[MODEL_EMBEDDING_SIZE];
    float max_error = 0.0f;
    model_timing_t model_timing;
    anomaly_result_t anomaly;
    int result = model_runner_compute(test_imu, test_audio, output, &model_timing);
    if (result != 0)
    {
        printf("[SMOKE][MODEL] FAIL code=%d\r\n", result);
        return result;
    }
    for (size_t index = 0U; index < MODEL_EMBEDDING_SIZE; ++index)
    {
        float error = fabsf(output[index] - expected_out[index]);
        if (error > max_error)
        {
            max_error = error;
        }
    }
    result = anomaly_detector_run(output, &anomaly);
    if (result != 0)
    {
        printf("[SMOKE][K2] FAIL code=%d\r\n", result);
        return result;
    }
    printf(
        "[SMOKE] MODEL=OK max_abs=%.8g TMAI_ms=%.3f K2_score=%.6f decision=%s\r\n",
        max_error,
        model_timing.latency_ms,
        anomaly.score,
        anomaly.anomaly ? "ANOMALY" : "NORMAL");
    return 0;
}

#else

int validation_smoke_run(void)
{
    return 0;
}

#endif
