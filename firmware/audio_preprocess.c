#include "audio_preprocess.h"

#include <math.h>
#include <string.h>

#include "audio_preprocess_constants.h"

static void swap_float(float *left, float *right)
{
    float temporary = *left;
    *left = *right;
    *right = temporary;
}

static void fft_1024(audio_preprocess_workspace_t *workspace)
{
    uint32_t j = 0U;

    /* In-place radix-2 DIT with an explicit bit reversal.  The fixed twiddle
     * table is generated in float32 and shared by host parity tests. */
    for (uint32_t i = 1U; i < AUDIO_FFT_SIZE; ++i)
    {
        uint32_t bit = AUDIO_FFT_SIZE >> 1U;
        while ((j & bit) != 0U)
        {
            j ^= bit;
            bit >>= 1U;
        }
        j ^= bit;
        if (i < j)
        {
            swap_float(&workspace->real[i], &workspace->real[j]);
            swap_float(&workspace->imag[i], &workspace->imag[j]);
        }
    }

    for (uint32_t length = 2U; length <= AUDIO_FFT_SIZE; length <<= 1U)
    {
        const uint32_t half = length >> 1U;
        const uint32_t twiddle_step = AUDIO_FFT_SIZE / length;
        for (uint32_t base = 0U; base < AUDIO_FFT_SIZE; base += length)
        {
            for (uint32_t offset = 0U; offset < half; ++offset)
            {
                const uint32_t twiddle = offset * twiddle_step;
                const float wr = audio_fft_twiddle_cos_1024[twiddle];
                const float wi = audio_fft_twiddle_sin_1024[twiddle];
                const uint32_t even = base + offset;
                const uint32_t odd = even + half;
                const float odd_real = workspace->real[odd];
                const float odd_imag = workspace->imag[odd];
                const float transformed_real = wr * odd_real - wi * odd_imag;
                const float transformed_imag = wi * odd_real + wr * odd_imag;
                const float even_real = workspace->real[even];
                const float even_imag = workspace->imag[even];

                workspace->real[even] = even_real + transformed_real;
                workspace->imag[even] = even_imag + transformed_imag;
                workspace->real[odd] = even_real - transformed_real;
                workspace->imag[odd] = even_imag - transformed_imag;
            }
        }
    }
}

int audio_preprocess_init(void)
{
    if ((audio_mel_offsets_40[0] != 0U) ||
        (audio_mel_offsets_40[AUDIO_MEL_BINS] !=
         AUDIO_MEL_NONZERO_COEFFICIENTS))
    {
        return APP_ERROR_PREPROCESS;
    }
    for (size_t band = 0U; band < AUDIO_MEL_BINS; ++band)
    {
        if (!(audio_normal_train_std_40[band] > 0.0f))
        {
            return APP_ERROR_PREPROCESS;
        }
    }
    return 0;
}

int audio_preprocess_frame(
    const int16_t pcm[AUDIO_FRAME_SAMPLES],
    float output[AUDIO_MEL_BINS],
    audio_preprocess_workspace_t *workspace,
    audio_preprocess_trace_t *trace)
{
    if ((pcm == NULL) || (output == NULL) || (workspace == NULL))
    {
        return APP_ERROR_PREPROCESS;
    }

    for (size_t sample = 0U; sample < AUDIO_FRAME_SAMPLES; ++sample)
    {
        const float converted = (float)pcm[sample] / 32768.0f;
        const float windowed = converted * audio_hann_640[sample];
        workspace->real[sample] = windowed;
        workspace->imag[sample] = 0.0f;
        if (trace != NULL)
        {
            trace->pcm[sample] = converted;
            trace->windowed[sample] = windowed;
        }
    }
    for (size_t sample = AUDIO_FRAME_SAMPLES; sample < AUDIO_FFT_SIZE; ++sample)
    {
        workspace->real[sample] = 0.0f;
        workspace->imag[sample] = 0.0f;
        if (trace != NULL)
        {
            trace->windowed[sample] = 0.0f;
        }
    }

    fft_1024(workspace);
    for (size_t bin = 0U; bin < AUDIO_POWER_BINS; ++bin)
    {
        const float real = workspace->real[bin];
        const float imag = workspace->imag[bin];
        const float power = real * real + imag * imag;
        workspace->power[bin] = power;
        if (trace != NULL)
        {
            trace->fft_real[bin] = real;
            trace->fft_imag[bin] = imag;
            trace->power[bin] = power;
        }
    }

    for (size_t band = 0U; band < AUDIO_MEL_BINS; ++band)
    {
        float mel_energy = 0.0f;
        const uint16_t begin = audio_mel_offsets_40[band];
        const uint16_t end = audio_mel_offsets_40[band + 1U];
        for (uint16_t item = begin; item < end; ++item)
        {
            mel_energy += workspace->power[audio_mel_bins_40[item]] *
                audio_mel_weights_40[item];
        }
        const float log_mel = logf(mel_energy + AUDIO_LOG_EPSILON);
        const float normalized =
            (log_mel - audio_normal_train_mean_40[band]) /
            audio_normal_train_std_40[band];
        output[band] = normalized;
        if (trace != NULL)
        {
            trace->mel[band] = mel_energy;
            trace->log_mel[band] = log_mel;
            trace->normalized[band] = normalized;
        }
    }

    return audio_preprocess_finite(output, AUDIO_MEL_BINS) ?
        0 : APP_ERROR_NONFINITE;
}

int audio_preprocess_window(
    const int16_t pcm[AUDIO_EVENT_SAMPLES],
    float output[AUDIO_FEATURE_FRAMES * AUDIO_MEL_BINS],
    audio_preprocess_workspace_t *workspace)
{
    if ((pcm == NULL) || (output == NULL) || (workspace == NULL))
    {
        return APP_ERROR_PREPROCESS;
    }
    for (size_t frame = 0U; frame < AUDIO_FEATURE_FRAMES; ++frame)
    {
        int result = audio_preprocess_frame(
            pcm + frame * AUDIO_HOP_SAMPLES,
            output + frame * AUDIO_MEL_BINS,
            workspace,
            NULL);
        if (result != 0)
        {
            return result;
        }
    }
    return 0;
}

int audio_preprocess_finite(const float *values, size_t count)
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
