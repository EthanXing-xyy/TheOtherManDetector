#include "microphone_capture.h"

#include <math.h>
#include <stddef.h>

#include "cybsp.h"
#include "cy_dma.h"
#include "cyhal.h"
#include "cy_pdl.h"
#include "cy_sysint.h"
#include "sample_window.h"
#include "timebase.h"

#define PDM_DATA_PIN (P10_5)
#define PDM_CLOCK_PIN (P10_4)
#define AUDIO_RING_MASK (AUDIO_RING_SAMPLES - 1U)
#define AUDIO_DMA_TRANSFERS_PER_BLOCK \
    (AUDIO_DMA_BLOCK_SAMPLES / AUDIO_DMA_TRANSFER_SAMPLES)
#define AUDIO_DMA_BUFFER_COUNT (2U)
#define AUDIO_DMA_ROUTE_INDEX (2U)
#define AUDIO_DMA_ROUTE_SELECT (1U)
#define AUDIO_DMA_INTERRUPT_MASK (1UL)

typedef struct
{
    uint32_t timestamp_us;
    uint64_t total_samples;
} audio_endpoint_t;

static cyhal_pdm_pcm_t pdm_pcm;
static cyhal_clock_t audio_clock;
static cyhal_clock_t audio_pll_clock;
CY_ALIGN(32) static int16_t audio_ring[AUDIO_RING_SAMPLES];
CY_ALIGN(32) static int16_t dma_buffers[AUDIO_DMA_BUFFER_COUNT][AUDIO_DMA_BLOCK_SAMPLES];
CY_ALIGN(32) static cy_stc_dma_descriptor_t dma_descriptors[AUDIO_DMA_DESCRIPTOR_COUNT];
static audio_endpoint_t audio_endpoints[AUDIO_ENDPOINT_COUNT];
static volatile uint64_t audio_sample_count;
static volatile uint32_t audio_block_count;
static volatile microphone_capture_fault_kind_t audio_fault_kind;
static volatile int32_t audio_fault_code;
static size_t endpoint_head;
static size_t endpoint_count;
static uint8_t completed_dma_buffer;
static uint32_t audio_rate_start_ms;
static uint64_t audio_rate_start_samples;
static bool audio_rate_observation_started;

static void append_endpoint(uint32_t timestamp_us, uint64_t total_samples)
{
    audio_endpoints[endpoint_head].timestamp_us = timestamp_us;
    audio_endpoints[endpoint_head].total_samples = total_samples;
    endpoint_head = (endpoint_head + 1U) % AUDIO_ENDPOINT_COUNT;
    if (endpoint_count < AUDIO_ENDPOINT_COUNT)
    {
        ++endpoint_count;
    }
}

static void set_fault_once(microphone_capture_fault_kind_t kind, int32_t code)
{
    uint32_t saved_interrupt_state = cyhal_system_critical_section_enter();
    if (audio_fault_kind == AUDIO_FAULT_NONE)
    {
        audio_fault_code = code;
        audio_fault_kind = kind;
    }
    cyhal_system_critical_section_exit(saved_interrupt_state);
}

static int32_t pdm_frequency_fix(void)
{
    const uint32_t clock_fields_mask =
        PDM_CLOCK_CTL_CLK_CLOCK_DIV_Msk |
        PDM_CLOCK_CTL_MCLKQ_CLOCK_DIV_Msk |
        PDM_CLOCK_CTL_CKO_CLOCK_DIV_Msk |
        PDM_CLOCK_CTL_SINC_RATE_Msk;
    uint32_t register_value =
        _VAL2FLD(PDM_CLOCK_CTL_CLK_CLOCK_DIV, AUDIO_PDM_CLOCK_DIV_STAGE_1 - 1U) |
        _VAL2FLD(PDM_CLOCK_CTL_MCLKQ_CLOCK_DIV, AUDIO_PDM_MCLK_DIV_STAGE_2 - 1U) |
        _VAL2FLD(PDM_CLOCK_CTL_CKO_CLOCK_DIV, AUDIO_PDM_OUTPUT_DIV_STAGE_3 - 1U) |
        _VAL2FLD(PDM_CLOCK_CTL_SINC_RATE, AUDIO_PDM_SINC_RATE);

    PDM0->CLOCK_CTL = register_value;
    return ((PDM0->CLOCK_CTL & clock_fields_mask) == register_value) ?
        0 : APP_ERROR_AUDIO_CLOCK;
}

static void pdm_callback(void *callback_arg, cyhal_pdm_pcm_event_t event)
{
    (void)callback_arg;

    if ((event & CYHAL_PDM_PCM_RX_OVERFLOW) != 0U)
    {
        set_fault_once(AUDIO_FAULT_RX_OVERFLOW, (int32_t)CYHAL_PDM_PCM_RX_OVERFLOW);
        Cy_DMA_Channel_Disable(DW1, AUDIO_DMA_CHANNEL);
    }
}

static void audio_dma_isr(void)
{
    cy_en_dma_intr_cause_t cause =
        Cy_DMA_Channel_GetStatus(DW1, AUDIO_DMA_CHANNEL);

    Cy_DMA_Channel_ClearInterrupt(DW1, AUDIO_DMA_CHANNEL);
    if (cause == CY_DMA_INTR_CAUSE_COMPLETION)
    {
        /* The cyclic chain is already filling the other ping-pong buffer.  Keep
         * this bounded copy in the ISR so capture remains independent of long
         * preprocessing/inference work in main; no HAL re-arm occurs here. */
        uint64_t block_start = audio_sample_count;
        const int16_t *completed = dma_buffers[completed_dma_buffer];
        for (size_t index = 0U; index < AUDIO_DMA_BLOCK_SAMPLES; ++index)
        {
            audio_ring[(uint32_t)(block_start + index) & AUDIO_RING_MASK] = completed[index];
        }
        audio_sample_count = block_start + AUDIO_DMA_BLOCK_SAMPLES;
        ++audio_block_count;
        append_endpoint(timebase_now_us(), audio_sample_count);
        completed_dma_buffer ^= 1U;
    }
    else
    {
        set_fault_once(AUDIO_FAULT_DMA, (int32_t)cause);
        Cy_DMA_Channel_Disable(DW1, AUDIO_DMA_CHANNEL);
    }
}

static void configure_dma_descriptors(void)
{
    /* Two 128-sample descriptors make each 256-sample logical block.  The
     * final descriptor of A chains directly to B (and B back to A), so the
     * PDM FIFO always has an armed DMA destination before the CPU is notified. */
    for (size_t index = 0U; index < AUDIO_DMA_DESCRIPTOR_COUNT; ++index)
    {
        const size_t buffer_index = index / AUDIO_DMA_TRANSFERS_PER_BLOCK;
        const size_t transfer_index = index % AUDIO_DMA_TRANSFERS_PER_BLOCK;
        cy_stc_dma_descriptor_t *descriptor = &dma_descriptors[index];

        Cy_DMA_Descriptor_DeInit(descriptor);
        Cy_DMA_Descriptor_SetDescriptorType(descriptor, CY_DMA_1D_TRANSFER);
        Cy_DMA_Descriptor_SetSrcAddress(
            descriptor, (const void *)(uintptr_t)&PDM0->RX_FIFO_RD);
        Cy_DMA_Descriptor_SetDstAddress(
            descriptor,
            &dma_buffers[buffer_index][transfer_index * AUDIO_DMA_TRANSFER_SAMPLES]);
        Cy_DMA_Descriptor_SetTriggerInType(descriptor, CY_DMA_DESCR);
        Cy_DMA_Descriptor_SetDataSize(descriptor, CY_DMA_HALFWORD);
        Cy_DMA_Descriptor_SetSrcTransferSize(descriptor, CY_DMA_TRANSFER_SIZE_WORD);
        Cy_DMA_Descriptor_SetDstTransferSize(descriptor, CY_DMA_TRANSFER_SIZE_DATA);
        Cy_DMA_Descriptor_SetXloopDataCount(descriptor, AUDIO_DMA_TRANSFER_SAMPLES);
        Cy_DMA_Descriptor_SetXloopSrcIncrement(descriptor, 0);
        Cy_DMA_Descriptor_SetXloopDstIncrement(descriptor, 1);
        Cy_DMA_Descriptor_SetInterruptType(
            descriptor,
            (transfer_index == (AUDIO_DMA_TRANSFERS_PER_BLOCK - 1U)) ?
                CY_DMA_DESCR : CY_DMA_DESCR_CHAIN);
        Cy_DMA_Descriptor_SetNextDescriptor(
            descriptor, &dma_descriptors[(index + 1U) % AUDIO_DMA_DESCRIPTOR_COUNT]);
    }
}

static int32_t configure_audio_dma(void)
{
    const cy_stc_sysint_t interrupt_config = {
        .intrSrc = cpuss_interrupts_dw1_26_IRQn,
        .intrPriority = AUDIO_DMA_ISR_PRIORITY,
    };
    cy_en_sysint_status_t interrupt_result;

    configure_dma_descriptors();
    Cy_DMA_Channel_Disable(DW1, AUDIO_DMA_CHANNEL);
    Cy_DMA_Channel_SetDescriptor(DW1, AUDIO_DMA_CHANNEL, &dma_descriptors[0]);
    Cy_DMA_Channel_SetPriority(DW1, AUDIO_DMA_CHANNEL, 0U);
    Cy_DMA_Channel_ClearInterrupt(DW1, AUDIO_DMA_CHANNEL);
    Cy_DMA_Channel_SetInterruptMask(
        DW1, AUDIO_DMA_CHANNEL, AUDIO_DMA_INTERRUPT_MASK);

    NVIC_DisableIRQ(cpuss_interrupts_dw1_26_IRQn);
    NVIC_ClearPendingIRQ(cpuss_interrupts_dw1_26_IRQn);
    interrupt_result = Cy_SysInt_Init(&interrupt_config, audio_dma_isr);
    if (interrupt_result != CY_SYSINT_SUCCESS)
    {
        set_fault_once(AUDIO_FAULT_DMA, (int32_t)interrupt_result);
        return (int32_t)interrupt_result;
    }
    return 0;
}

int32_t microphone_capture_init(void)
{
    const cyhal_pdm_pcm_cfg_t pdm_config = {
        .sample_rate = AUDIO_SAMPLE_RATE_HZ,
        .decimation_rate = AUDIO_DECIMATION_RATE,
        .mode = CYHAL_PDM_PCM_MODE_LEFT,
        .word_length = AUDIO_BITS_PER_SAMPLE,
        .left_gain = AUDIO_MIC_GAIN_HALF_DB_STEPS,
        .right_gain = AUDIO_MIC_GAIN_HALF_DB_STEPS,
    };

    audio_sample_count = 0U;
    audio_block_count = 0U;
    audio_fault_kind = AUDIO_FAULT_NONE;
    audio_fault_code = 0;
    endpoint_head = 0U;
    endpoint_count = 0U;
    completed_dma_buffer = 0U;
    audio_rate_start_ms = 0U;
    audio_rate_start_samples = 0U;
    audio_rate_observation_started = false;

    cy_rslt_t result = cyhal_clock_reserve(&audio_pll_clock, &CYHAL_CLOCK_PLL[1]);
    if (result != CY_RSLT_SUCCESS)
    {
        set_fault_once(AUDIO_FAULT_HAL, (int32_t)result);
        return (int32_t)result;
    }
    result = cyhal_clock_set_frequency(&audio_pll_clock, AUDIO_SYSTEM_CLOCK_HZ, NULL);
    if (result != CY_RSLT_SUCCESS)
    {
        set_fault_once(AUDIO_FAULT_HAL, (int32_t)result);
        return (int32_t)result;
    }
    result = cyhal_clock_set_enabled(&audio_pll_clock, true, true);
    if (result != CY_RSLT_SUCCESS)
    {
        set_fault_once(AUDIO_FAULT_HAL, (int32_t)result);
        return (int32_t)result;
    }
    result = cyhal_clock_reserve(&audio_clock, &CYHAL_CLOCK_HF[1]);
    if (result != CY_RSLT_SUCCESS)
    {
        set_fault_once(AUDIO_FAULT_HAL, (int32_t)result);
        return (int32_t)result;
    }
    result = cyhal_clock_set_source(&audio_clock, &audio_pll_clock);
    if (result != CY_RSLT_SUCCESS)
    {
        set_fault_once(AUDIO_FAULT_HAL, (int32_t)result);
        return (int32_t)result;
    }
    result = cyhal_clock_set_enabled(&audio_clock, true, true);
    if (result != CY_RSLT_SUCCESS)
    {
        set_fault_once(AUDIO_FAULT_HAL, (int32_t)result);
        return (int32_t)result;
    }
    result = cyhal_pdm_pcm_init(
        &pdm_pcm, PDM_DATA_PIN, PDM_CLOCK_PIN, &audio_clock, &pdm_config);
    if (result != CY_RSLT_SUCCESS)
    {
        set_fault_once(AUDIO_FAULT_HAL, (int32_t)result);
        return (int32_t)result;
    }

    cyhal_pdm_pcm_register_callback(&pdm_pcm, pdm_callback, NULL);
    cyhal_pdm_pcm_enable_event(
        &pdm_pcm,
        CYHAL_PDM_PCM_RX_OVERFLOW,
        AUDIO_DMA_ISR_PRIORITY,
        true);
    return configure_audio_dma();
}

int32_t microphone_capture_start(void)
{
    int32_t clock_result = pdm_frequency_fix();
    cy_rslt_t result;

    if (clock_result != 0)
    {
        set_fault_once(AUDIO_FAULT_HAL, clock_result);
        return clock_result;
    }

    append_endpoint(timebase_now_us(), 0U);
    PDM0->TR_CTL = _VAL2FLD(PDM_TR_CTL_RX_REQ_EN, 1U);
    PERI_TR_1TO1_GR4->TR_CTL[AUDIO_DMA_ROUTE_INDEX] =
        _VAL2FLD(PERI_TR_1TO1_GR_V2_TR_CTL_TR_SEL, AUDIO_DMA_ROUTE_SELECT);
    Cy_DMA_Channel_ClearInterrupt(DW1, AUDIO_DMA_CHANNEL);
    NVIC_ClearPendingIRQ(cpuss_interrupts_dw1_26_IRQn);
    NVIC_EnableIRQ(cpuss_interrupts_dw1_26_IRQn);
    Cy_DMA_Channel_Enable(DW1, AUDIO_DMA_CHANNEL);
    Cy_DMA_Enable(DW1);

    result = cyhal_pdm_pcm_start(&pdm_pcm);
    if (result != CY_RSLT_SUCCESS)
    {
        Cy_DMA_Channel_Disable(DW1, AUDIO_DMA_CHANNEL);
        NVIC_DisableIRQ(cpuss_interrupts_dw1_26_IRQn);
        set_fault_once(AUDIO_FAULT_HAL, (int32_t)result);
        return (int32_t)result;
    }

    audio_rate_start_ms = timebase_now_ms();
    audio_rate_start_samples = microphone_capture_total_samples();
    audio_rate_observation_started = true;
    return 0;
}

bool microphone_capture_config_readback(microphone_capture_config_readback_t *readback)
{
    uint32_t clock_ctl;
    cy_en_pdm_pcm_gain_t gain;

    if (readback == NULL)
    {
        return false;
    }

    clock_ctl = PDM0->CLOCK_CTL;
    gain = Cy_PDM_PCM_GetGain(PDM0, CY_PDM_PCM_CHAN_LEFT);
    if ((uint32_t)gain > (uint32_t)CY_PDM_PCM_GAIN_10_5_DB)
    {
        return false;
    }

    readback->pll_hz = cyhal_clock_get_frequency(&audio_pll_clock);
    readback->audio_clock_hz = cyhal_clock_get_frequency(&audio_clock);
    readback->pdm_clock_ctl = clock_ctl;
    readback->clock_div_stage_1 =
        _FLD2VAL(PDM_CLOCK_CTL_CLK_CLOCK_DIV, clock_ctl) + 1U;
    readback->mclk_div_stage_2 =
        _FLD2VAL(PDM_CLOCK_CTL_MCLKQ_CLOCK_DIV, clock_ctl) + 1U;
    readback->output_div_stage_3 =
        _FLD2VAL(PDM_CLOCK_CTL_CKO_CLOCK_DIV, clock_ctl) + 1U;
    readback->sinc_rate = _FLD2VAL(PDM_CLOCK_CTL_SINC_RATE, clock_ctl);
    readback->hardware_gain_enum = (uint32_t)gain;
    readback->hardware_gain_db = -12.0f + (1.5f * (float)gain);
    return true;
}

bool microphone_capture_rate_stats(microphone_capture_rate_stats_t *stats)
{
    uint32_t end_ms;
    uint64_t end_samples;
    uint32_t saved_interrupt_state;
    float configured_hz = (float)AUDIO_SAMPLE_RATE_HZ;

    if ((stats == NULL) || !audio_rate_observation_started)
    {
        return false;
    }

    saved_interrupt_state = cyhal_system_critical_section_enter();
    end_ms = timebase_now_ms();
    end_samples = audio_sample_count;
    cyhal_system_critical_section_exit(saved_interrupt_state);

    stats->elapsed_ms = timebase_elapsed_ms(audio_rate_start_ms, end_ms);
    stats->delta_samples = end_samples - audio_rate_start_samples;
    if (stats->elapsed_ms == 0U)
    {
        stats->measured_hz = 0.0f;
        stats->error_percent = 0.0f;
        return false;
    }
    stats->measured_hz =
        ((float)stats->delta_samples * 1000.0f) / (float)stats->elapsed_ms;
    stats->error_percent =
        (fabsf(stats->measured_hz - configured_hz) * 100.0f) / configured_hz;
    return true;
}

uint64_t microphone_capture_total_samples(void)
{
    uint64_t total;
    uint32_t saved_interrupt_state = cyhal_system_critical_section_enter();
    total = audio_sample_count;
    cyhal_system_critical_section_exit(saved_interrupt_state);
    return total;
}

uint32_t microphone_capture_completed_blocks(void)
{
    uint32_t blocks;
    uint32_t saved_interrupt_state = cyhal_system_critical_section_enter();
    blocks = audio_block_count;
    cyhal_system_critical_section_exit(saved_interrupt_state);
    return blocks;
}

bool microphone_capture_history_ready(void)
{
    return !microphone_capture_faulted() &&
        (microphone_capture_total_samples() >= AUDIO_PRETRIGGER_SAMPLES);
}

bool microphone_capture_timestamp_to_sample(uint32_t timestamp_us, uint64_t *sample_index)
{
    bool found = false;
    uint32_t saved_interrupt_state;

    if ((sample_index == NULL) || microphone_capture_faulted())
    {
        return false;
    }

    saved_interrupt_state = cyhal_system_critical_section_enter();
    if (endpoint_count >= 2U)
    {
        size_t oldest =
            (endpoint_head + AUDIO_ENDPOINT_COUNT - endpoint_count) % AUDIO_ENDPOINT_COUNT;
        for (size_t index = 0U; index + 1U < endpoint_count; ++index)
        {
            const audio_endpoint_t *before =
                &audio_endpoints[(oldest + index) % AUDIO_ENDPOINT_COUNT];
            const audio_endpoint_t *after =
                &audio_endpoints[(oldest + index + 1U) % AUDIO_ENDPOINT_COUNT];
            uint32_t span_us = after->timestamp_us - before->timestamp_us;
            uint32_t offset_us = timestamp_us - before->timestamp_us;
            if ((span_us != 0U) && (offset_us <= span_us))
            {
                uint64_t span_samples = after->total_samples - before->total_samples;
                *sample_index = before->total_samples +
                    ((span_samples * offset_us + (span_us / 2U)) / span_us);
                found = true;
                break;
            }
        }
    }
    cyhal_system_critical_section_exit(saved_interrupt_state);
    return found;
}

bool microphone_capture_target_available(uint64_t t0_sample_index)
{
    return !microphone_capture_faulted() &&
        (microphone_capture_total_samples() >= t0_sample_index + AUDIO_POSTTRIGGER_SAMPLES);
}

bool microphone_capture_range_available(uint64_t first_sample, size_t count)
{
    uint64_t total = microphone_capture_total_samples();
    uint64_t oldest = (total > AUDIO_RING_SAMPLES) ?
        total - AUDIO_RING_SAMPLES : 0U;

    if (first_sample > UINT64_MAX - (uint64_t)count)
    {
        return false;
    }
    return !microphone_capture_faulted() &&
        (first_sample >= oldest) &&
        (first_sample + (uint64_t)count <= total);
}

bool microphone_capture_copy_range(
    uint64_t first_sample, size_t count, int16_t *destination)
{
    uint64_t total;
    uint32_t saved_interrupt_state;
    bool copied;

    if ((destination == NULL) || microphone_capture_faulted())
    {
        return false;
    }
    saved_interrupt_state = cyhal_system_critical_section_enter();
    total = audio_sample_count;
    copied = sample_window_copy_i16_ring(
        audio_ring,
        AUDIO_RING_SAMPLES,
        total,
        first_sample,
        count,
        destination);
    cyhal_system_critical_section_exit(saved_interrupt_state);
    return copied && !microphone_capture_faulted();
}

bool microphone_capture_snapshot_event(uint64_t t0_sample_index, int16_t *destination)
{
    uint64_t start_sample;

    if ((destination == NULL) || microphone_capture_faulted() ||
        (t0_sample_index < AUDIO_PRETRIGGER_SAMPLES))
    {
        return false;
    }
    start_sample = t0_sample_index - AUDIO_PRETRIGGER_SAMPLES;
    return microphone_capture_copy_range(start_sample, AUDIO_EVENT_SAMPLES, destination);
}

bool microphone_capture_faulted(void)
{
    return audio_fault_kind != AUDIO_FAULT_NONE;
}

int32_t microphone_capture_fault_code(void)
{
    return microphone_capture_fault_info().code;
}

microphone_capture_fault_info_t microphone_capture_fault_info(void)
{
    microphone_capture_fault_info_t fault;
    uint32_t saved_interrupt_state = cyhal_system_critical_section_enter();
    fault.kind = audio_fault_kind;
    fault.code = audio_fault_code;
    cyhal_system_critical_section_exit(saved_interrupt_state);
    return fault;
}

const char *microphone_capture_fault_name(microphone_capture_fault_kind_t kind)
{
    switch (kind)
    {
        case AUDIO_FAULT_NONE:
            return "NONE";
        case AUDIO_FAULT_RX_OVERFLOW:
            return "RX_OVERFLOW";
        case AUDIO_FAULT_ASYNC_REARM:
            return "ASYNC_REARM";
        case AUDIO_FAULT_HAL:
            return "HAL";
        case AUDIO_FAULT_DMA:
            return "DMA";
        default:
            return "UNKNOWN";
    }
}

#if ENABLE_DEPLOYMENT_SELF_TEST
bool microphone_capture_self_test(void)
{
    static int16_t snapshot[AUDIO_EVENT_SAMPLES];
    const uint32_t before_ms = UINT32_MAX - 8U;
    const uint32_t after_ms = 7U;
    const uint64_t before_samples = 1000U;
    const uint64_t after_samples = 1256U;
    const uint64_t expected_midpoint = 1128U;
    const uint64_t t0_sample = 70000U;
    const uint64_t start_sample = t0_sample - AUDIO_PRETRIGGER_SAMPLES;
    const uint64_t end_sample = t0_sample + AUDIO_POSTTRIGGER_SAMPLES;
    uint64_t mapped_sample = 0U;

    audio_fault_kind = AUDIO_FAULT_NONE;
    audio_fault_code = 0;
    audio_sample_count = 0U;
    audio_block_count = 0U;
    endpoint_head = 0U;
    endpoint_count = 0U;
    append_endpoint(before_ms, before_samples);
    append_endpoint(after_ms, after_samples);
    if (!microphone_capture_timestamp_to_sample(UINT32_MAX, &mapped_sample) ||
        (mapped_sample != expected_midpoint))
    {
        return false;
    }

    for (uint64_t sample = start_sample; sample < end_sample; ++sample)
    {
        audio_ring[(uint32_t)sample & AUDIO_RING_MASK] =
            (int16_t)((int32_t)(sample % 30001U) - 15000);
    }
    audio_sample_count = end_sample;
    if (!microphone_capture_target_available(t0_sample) ||
        !microphone_capture_snapshot_event(t0_sample, snapshot))
    {
        return false;
    }
    for (size_t index = 0U; index < AUDIO_EVENT_SAMPLES; ++index)
    {
        int16_t expected =
            (int16_t)((int32_t)((start_sample + index) % 30001U) - 15000);
        if (snapshot[index] != expected)
        {
            return false;
        }
    }

    audio_sample_count = end_sample + (AUDIO_RING_SAMPLES - AUDIO_EVENT_SAMPLES) + 1U;
    if (microphone_capture_snapshot_event(t0_sample, snapshot))
    {
        return false;
    }

    audio_fault_kind = AUDIO_FAULT_DMA;
    audio_fault_code = APP_ERROR_AUDIO_DMA;
    if (microphone_capture_history_ready() || microphone_capture_target_available(t0_sample) ||
        microphone_capture_timestamp_to_sample(UINT32_MAX, &mapped_sample) ||
        microphone_capture_snapshot_event(t0_sample, snapshot))
    {
        return false;
    }
    if ((microphone_capture_fault_info().kind != AUDIO_FAULT_DMA) ||
        (microphone_capture_fault_code() != APP_ERROR_AUDIO_DMA) ||
        (microphone_capture_fault_name(AUDIO_FAULT_RX_OVERFLOW)[0] != 'R'))
    {
        return false;
    }
    audio_fault_kind = AUDIO_FAULT_NONE;
    audio_fault_code = 0;
    return true;
}
#endif
