#pragma once

#include <stddef.h>
#include <stdint.h>

#include "app_config.h"

typedef struct
{
    float real[AUDIO_FFT_SIZE];
    float imag[AUDIO_FFT_SIZE];
    float power[AUDIO_POWER_BINS];
} audio_preprocess_workspace_t;

/* Optional host/debug trace.  Firmware passes NULL, so this consumes no
 * static SRAM in production. */
typedef struct
{
    float pcm[AUDIO_FRAME_SAMPLES];
    float windowed[AUDIO_FFT_SIZE];
    float fft_real[AUDIO_POWER_BINS];
    float fft_imag[AUDIO_POWER_BINS];
    float power[AUDIO_POWER_BINS];
    float mel[AUDIO_MEL_BINS];
    float log_mel[AUDIO_MEL_BINS];
    float normalized[AUDIO_MEL_BINS];
} audio_preprocess_trace_t;

int audio_preprocess_init(void);
int audio_preprocess_frame(
    const int16_t pcm[AUDIO_FRAME_SAMPLES],
    float output[AUDIO_MEL_BINS],
    audio_preprocess_workspace_t *workspace,
    audio_preprocess_trace_t *trace);
int audio_preprocess_window(
    const int16_t pcm[AUDIO_EVENT_SAMPLES],
    float output[AUDIO_FEATURE_FRAMES * AUDIO_MEL_BINS],
    audio_preprocess_workspace_t *workspace);
int audio_preprocess_finite(
    const float *values, size_t count);
