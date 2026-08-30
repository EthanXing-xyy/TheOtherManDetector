#include "model_runner.h"

#include <stdbool.h>

#include "cy_pdl.h"
#include "cy_utils.h"
#include "model/embedding_model.h"

static uint32_t profiler_clock_hz;
static bool profiler_enabled;

static void profiler_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    profiler_enabled = (DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0U;
    profiler_clock_hz = SystemCoreClock;
}

int model_runner_init(void)
{
    profiler_init();
    return TMAI_init();
}

int model_runner_reset(void)
{
    return (TMAI_soft_reset() == TMAI_RET_SUCCESS) ?
        0 : APP_ERROR_MODEL_RESET;
}

uint32_t model_runner_cycle_now(void)
{
    return profiler_enabled ? DWT->CYCCNT : 0U;
}

float model_runner_cycles_to_ms(uint32_t cycles)
{
    return (profiler_enabled && (profiler_clock_hz != 0U)) ?
        ((float)cycles * 1000.0f) / (float)profiler_clock_hz : -1.0f;
}

int model_runner_compute(
    const float imu[MODEL_IMU_FRAMES * MODEL_IMU_CHANNELS],
    const float audio[MODEL_AUDIO_FRAMES * MODEL_AUDIO_MELS],
    float embedding[MODEL_EMBEDDING_SIZE],
    model_timing_t *timing)
{
    uint32_t start_cycles;
    uint32_t end_cycles;
    int result;

    result = model_runner_reset();
    if (result != 0)
    {
        return result;
    }
    __DSB();
    __ISB();
    start_cycles = model_runner_cycle_now();
    result = TMAI_compute(imu, audio, embedding); /* IMU first, Audio second. */
    end_cycles = model_runner_cycle_now();
    __DSB();
    __ISB();
    if (timing != NULL)
    {
        timing->cycles = end_cycles - start_cycles;
        timing->latency_ms = model_runner_cycles_to_ms(timing->cycles);
    }
    return (result == TMAI_RET_SUCCESS) ? 0 : APP_ERROR_MODEL_COMPUTE;
}
