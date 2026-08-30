#pragma once

#include <stdint.h>

#include "app_config.h"

typedef struct
{
    float score;
    float distances[2];
    float threshold;
    int nearest_prototype;
    int anomaly;
    uint32_t cycles;
    float latency_ms;
} anomaly_result_t;

int anomaly_detector_init(void);
int anomaly_detector_run(
    const float embedding[MODEL_EMBEDDING_SIZE], anomaly_result_t *result);
