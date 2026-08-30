#pragma once

#include <stdint.h>

/* Build-time diagnostics.  Production defaults are all quiet. */
#ifndef ENABLE_TRIGGER_DEBUG
#define ENABLE_TRIGGER_DEBUG (0)
#endif
#ifndef ENABLE_DSP_DEBUG
#define ENABLE_DSP_DEBUG (0)
#endif
#ifndef ENABLE_SAMPLE_WINDOW_DEBUG
#define ENABLE_SAMPLE_WINDOW_DEBUG (0)
#endif
#ifndef ENABLE_VALIDATION_SMOKE_TEST
#define ENABLE_VALIDATION_SMOKE_TEST (0)
#endif
#ifndef REAL_SENSOR_MODE
#define REAL_SENSOR_MODE (1)
#endif

/* Training-time common event geometry: [-2.5 s, +4.7 s). */
#define EVENT_PRETRIGGER_US (2500000U)
#define EVENT_POSTTRIGGER_US (4700000U)
#define EVENT_DURATION_US (7200000U)

/* BMI270 contract recovered from the Deepcraft stream and training artifacts. */
#define IMU_SAMPLE_RATE_HZ (100U)
#define IMU_SAMPLE_PERIOD_US (10000U)
#define IMU_CHANNELS (6U)
#define IMU_ACCEL_AXES (3U)
#define IMU_ACCEL_RANGE_G (8U)
#define IMU_GYRO_RANGE_DPS (500U)
#define IMU_PRETRIGGER_FRAMES (250U)
#define IMU_POSTTRIGGER_FRAMES (470U)
#define IMU_EVENT_FRAMES (IMU_PRETRIGGER_FRAMES + IMU_POSTTRIGGER_FRAMES)
#define IMU_HISTORY_FRAMES (1024U)
#define IMU_I2C_FREQUENCY_HZ (400000U)
#define IMU_ISR_PRIORITY (4U)

/* MailmanDetectorDeploy deterministic door detector, centralized for mounting. */
#define DOOR_DIRECTION_AXIS (1U) /* training/door mounting: Ay */
#define OPEN_DIRECTION_SIGN (-1.0f)
#define OPEN_DIRECTION_MAGNITUDE_MPS2 (0.05f)
#define CLOSE_DIRECTION_MAGNITUDE_MPS2 (0.12f)
#define OPEN_DIRECTION_THRESHOLD \
    (OPEN_DIRECTION_SIGN * OPEN_DIRECTION_MAGNITUDE_MPS2)
#define CLOSE_DIRECTION_THRESHOLD \
    (-OPEN_DIRECTION_SIGN * CLOSE_DIRECTION_MAGNITUDE_MPS2)
#define OPEN_CONSECUTIVE_FRAMES (5U)
#define CLOSE_CONSECUTIVE_FRAMES (5U)
#define IMU_BASELINE_FRAMES (50U)
#define MOTION_THRESHOLD_MPS2 (0.05f)
#define OPEN_SETTLE_WINDOW_FRAMES (50U)
#define OPEN_SETTLE_ACC_VAR_THRESHOLD (0.00016f)
#define OPEN_SETTLE_GYRO_VAR_THRESHOLD (0.025f)
#define DOOR_REARM_DELAY_MS (5000U)
#define DOOR_REARM_DELAY_US (DOOR_REARM_DELAY_MS * 1000U)

/* PDM/PCM path. Gain is requested in HAL half-dB units.  The CAT1 PDL
 * quantizes request 10 to hardware enum 11, i.e. actual +4.5 dB. */
#define AUDIO_SAMPLE_RATE_HZ (16000U)
#define AUDIO_CHANNELS (1U)
#define AUDIO_SYSTEM_CLOCK_HZ (24576000U)
#define AUDIO_DECIMATION_RATE (64U)
#define AUDIO_BITS_PER_SAMPLE (16U)
#define AUDIO_GRAPH_GAIN_DB (5)
#define AUDIO_MIC_GAIN_HALF_DB_STEPS (10)
#define AUDIO_EXPECTED_HW_GAIN_ENUM (11U)
#define AUDIO_EXPECTED_HW_GAIN_DB (4.5f)
#define AUDIO_PDM_CLOCK_DIV_STAGE_1 (2U)
#define AUDIO_PDM_MCLK_DIV_STAGE_2 (1U)
#define AUDIO_PDM_OUTPUT_DIV_STAGE_3 (8U)
#define AUDIO_PDM_SINC_RATE \
    (AUDIO_SYSTEM_CLOCK_HZ / \
     (AUDIO_PDM_CLOCK_DIV_STAGE_1 * AUDIO_PDM_MCLK_DIV_STAGE_2 * \
      AUDIO_PDM_OUTPUT_DIV_STAGE_3 * 2U * AUDIO_SAMPLE_RATE_HZ))
#define AUDIO_DMA_BLOCK_SAMPLES (256U)
#define AUDIO_DMA_TRANSFER_SAMPLES (128U)
#define AUDIO_DMA_DESCRIPTOR_COUNT (4U)
#define AUDIO_DMA_CHANNEL (26U)
#define AUDIO_DMA_ISR_PRIORITY (3U)
#define AUDIO_RING_SAMPLES (131072U)
#define AUDIO_ENDPOINT_COUNT (128U)
#define AUDIO_PRETRIGGER_SAMPLES (40000U)
#define AUDIO_POSTTRIGGER_SAMPLES (75200U)
#define AUDIO_EVENT_SAMPLES (AUDIO_PRETRIGGER_SAMPLES + AUDIO_POSTTRIGGER_SAMPLES)

/* Exact training audio frontend. */
#define AUDIO_FRAME_SAMPLES (640U)
#define AUDIO_HOP_SAMPLES (640U)
#define AUDIO_FFT_SIZE (1024U)
#define AUDIO_POWER_BINS (513U)
#define AUDIO_MEL_BINS (40U)
#define AUDIO_FEATURE_FRAMES (180U)
#define AUDIO_FMIN_HZ (0U)
#define AUDIO_FMAX_HZ (7000U)
#define AUDIO_LOG_EPSILON (1.0e-6f)

/* Generated embedding model and independently calibrated K=2 anomaly head. */
#define MODEL_IMU_FRAMES (720U)
#define MODEL_IMU_CHANNELS (6U)
#define MODEL_AUDIO_FRAMES (180U)
#define MODEL_AUDIO_MELS (40U)
#define MODEL_EMBEDDING_SIZE (32U)
/*
 * The base anomaly threshold is derived from the upper boundary of
 * Mahalanobis scores on the Normal Validation set.
 *
 * A fixed 5% deployment safety factor provides additional tolerance against
 * minor sensor variation, installation deviation, and environmental domain
 * shift. This factor only adjusts the deployment decision boundary; it does
 * not modify the neural network, feature embedding, K=2 normal-space model,
 * scaler, prototypes, covariance/precision matrix, or Mahalanobis score.
 */
#define BASE_ANOMALY_THRESHOLD (5.692456722f)
#define ANOMALY_THRESHOLD_SAFETY_FACTOR (1.05f)
#define DEPLOYMENT_ANOMALY_THRESHOLD \
    (BASE_ANOMALY_THRESHOLD * ANOMALY_THRESHOLD_SAFETY_FACTOR)

/* Startup/runtime health gates. */
#define BOOT_SENSOR_TIMEOUT_MS (5000U)
#define BOOT_MIN_IMU_SAMPLES (250U)
#define BOOT_MIN_AUDIO_SAMPLES (40000U)
#define AUDIO_CLOCK_READBACK_TOLERANCE_PPM (10000U)
#define SENSOR_RATE_WARNING_PERCENT (1.0f)

/* Application-owned negative errors; positive/non-zero HAL/Bosch results are
 * preserved verbatim where possible. */
#define APP_ERROR_AUDIO_TIMESTAMP (-1001)
#define APP_ERROR_IMU_HISTORY (-1002)
#define APP_ERROR_AUDIO_DMA (-1003)
#define APP_ERROR_AUDIO_RING_OVERWRITE (-1004)
#define APP_ERROR_PREPROCESS (-1005)
#define APP_ERROR_IMU_SENSOR (-1006)
#define APP_ERROR_AUDIO_CLOCK (-1007)
#define APP_ERROR_MODEL_RESET (-1008)
#define APP_ERROR_MODEL_COMPUTE (-1009)
#define APP_ERROR_IMU_CONFIG (-1010)
#define APP_ERROR_IMU_RESOLUTION (-1011)
#define APP_ERROR_WINDOW_COUNT (-1012)
#define APP_ERROR_AUDIO_CONFIG (-1013)
#define APP_ERROR_NONFINITE (-1014)
#define APP_ERROR_K2 (-1015)

_Static_assert(IMU_SAMPLE_RATE_HZ == 100U, "training requires 100 Hz IMU");
_Static_assert(IMU_ACCEL_RANGE_G == 8U, "training requires +/-8 g");
_Static_assert(IMU_GYRO_RANGE_DPS == 500U, "training requires +/-500 dps");
_Static_assert(IMU_EVENT_FRAMES == 720U, "event must contain exactly 720 IMU samples");
_Static_assert(AUDIO_SAMPLE_RATE_HZ == 16000U, "training requires 16 kHz audio");
_Static_assert(AUDIO_CHANNELS == 1U, "training requires mono audio");
_Static_assert(AUDIO_EVENT_SAMPLES == 115200U, "event must contain exactly 115200 PCM samples");
_Static_assert(AUDIO_FRAME_SAMPLES == AUDIO_HOP_SAMPLES, "training has no overlap");
_Static_assert(AUDIO_FEATURE_FRAMES * AUDIO_FRAME_SAMPLES == AUDIO_EVENT_SAMPLES,
               "180 x 640 must cover the exact event window");
_Static_assert(AUDIO_PRETRIGGER_SAMPLES == (62U * 640U + 320U),
               "t0 must remain at frame 62 sample 320");
_Static_assert((AUDIO_RING_SAMPLES & (AUDIO_RING_SAMPLES - 1U)) == 0U,
               "audio ring must be a power of two");
_Static_assert((IMU_HISTORY_FRAMES & (IMU_HISTORY_FRAMES - 1U)) == 0U,
               "IMU ring must be a power of two");
_Static_assert(AUDIO_RING_SAMPLES > AUDIO_EVENT_SAMPLES,
               "audio ring must exceed one event; frames are consumed streaming");
_Static_assert(AUDIO_MIC_GAIN_HALF_DB_STEPS == 2 * AUDIO_GRAPH_GAIN_DB,
               "Gain 5 maps to ten HAL half-dB steps");
_Static_assert(AUDIO_PDM_SINC_RATE == 48U, "PDM clock path requires SINC=48");
_Static_assert(MODEL_IMU_FRAMES == IMU_EVENT_FRAMES, "model IMU geometry mismatch");
_Static_assert(MODEL_AUDIO_FRAMES == AUDIO_FEATURE_FRAMES, "model audio geometry mismatch");
_Static_assert(OPEN_DIRECTION_THRESHOLD < 0.0f, "reference mounting opens on negative Ay");
_Static_assert(CLOSE_DIRECTION_THRESHOLD > 0.0f, "close must be opposite to open");
