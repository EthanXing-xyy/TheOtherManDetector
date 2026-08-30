#include "anomaly_detector.h"

#include <math.h>
#include <stddef.h>

#include "model_runner.h"
#include "model/mahalanobis_detector.h"

int anomaly_detector_init(void)
{
    return 0;
}

int anomaly_detector_run(
    const float embedding[MODEL_EMBEDDING_SIZE], anomaly_result_t *result)
{
    uint32_t start_cycles;
    uint32_t end_cycles;

    if ((embedding == NULL) || (result == NULL))
    {
        return APP_ERROR_K2;
    }
    for (size_t index = 0U; index < MODEL_EMBEDDING_SIZE; ++index)
    {
        if (!isfinite(embedding[index]))
        {
            return APP_ERROR_NONFINITE;
        }
    }

    start_cycles = model_runner_cycle_now();
    result->score = mahalanobis_anomaly_score(
        embedding, result->distances, &result->nearest_prototype);
    end_cycles = model_runner_cycle_now();
    if (!isfinite(result->score) || !isfinite(result->distances[0]) ||
        !isfinite(result->distances[1]))
    {
        return APP_ERROR_NONFINITE;
    }
    result->threshold = DEPLOYMENT_ANOMALY_THRESHOLD;
    result->anomaly = result->score > result->threshold;
    result->cycles = end_cycles - start_cycles;
    result->latency_ms = model_runner_cycles_to_ms(result->cycles);
    return 0;
}
