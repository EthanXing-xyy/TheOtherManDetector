#include "mahalanobis_detector.h"

#include <math.h>
#include <stddef.h>

static void standardize_embedding(
    const float embedding[MAHALANOBIS_EMBEDDING_DIM],
    float standardized[MAHALANOBIS_EMBEDDING_DIM]) {
    int index;
    for (index = 0; index < MAHALANOBIS_EMBEDDING_DIM; ++index) {
        standardized[index] =
            (embedding[index] - MAHALANOBIS_EMBEDDING_MEAN[index]) /
            MAHALANOBIS_EMBEDDING_SCALE[index];
    }
}

static float mahalanobis_distance(
    const float standardized[MAHALANOBIS_EMBEDDING_DIM],
    const float prototype[MAHALANOBIS_EMBEDDING_DIM]) {
    float difference[MAHALANOBIS_EMBEDDING_DIM];
    float squared_distance = 0.0f;
    int row;
    int column;
    for (row = 0; row < MAHALANOBIS_EMBEDDING_DIM; ++row) {
        difference[row] = standardized[row] - prototype[row];
    }
    for (row = 0; row < MAHALANOBIS_EMBEDDING_DIM; ++row) {
        float row_product = 0.0f;
        for (column = 0; column < MAHALANOBIS_EMBEDDING_DIM; ++column) {
            row_product += MAHALANOBIS_PRECISION[row][column] * difference[column];
        }
        squared_distance += difference[row] * row_product;
    }
    return sqrtf(fmaxf(squared_distance, 0.0f));
}

float mahalanobis_anomaly_score(
    const float embedding[MAHALANOBIS_EMBEDDING_DIM],
    float distances[MAHALANOBIS_PROTOTYPE_COUNT],
    int *nearest_prototype) {
    float standardized[MAHALANOBIS_EMBEDDING_DIM];
    float local_distances[MAHALANOBIS_PROTOTYPE_COUNT];
    int best_index;
    standardize_embedding(embedding, standardized);
    local_distances[0] = mahalanobis_distance(standardized, MAHALANOBIS_PROTOTYPES[0]);
    local_distances[1] = mahalanobis_distance(standardized, MAHALANOBIS_PROTOTYPES[1]);
    best_index = local_distances[1] < local_distances[0] ? 1 : 0;
    if (distances != NULL) {
        distances[0] = local_distances[0];
        distances[1] = local_distances[1];
    }
    if (nearest_prototype != NULL) {
        *nearest_prototype = best_index;
    }
    return local_distances[best_index];
}

int mahalanobis_predict(
    const float embedding[MAHALANOBIS_EMBEDDING_DIM],
    float *score,
    int *nearest_prototype) {
    const float value = mahalanobis_anomaly_score(embedding, NULL, nearest_prototype);
    if (score != NULL) {
        *score = value;
    }
    return value > MAHALANOBIS_BASE_THRESHOLD ? 1 : 0;
}
