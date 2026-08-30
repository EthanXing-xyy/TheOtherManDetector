#include "timebase.h"

#include <limits.h>

#include "cyhal.h"

static cyhal_timer_t monotonic_timer;

int32_t timebase_init(void)
{
    const cyhal_timer_cfg_t config = {
        .is_continuous = true,
        .direction = CYHAL_TIMER_DIR_UP,
        .is_compare = false,
        .period = UINT32_MAX,
        .compare_value = 0U,
        .value = 0U,
    };
    cy_rslt_t result = cyhal_timer_init(&monotonic_timer, NC, NULL);
    if (result != CY_RSLT_SUCCESS)
    {
        return (int32_t)result;
    }
    result = cyhal_timer_set_frequency(&monotonic_timer, 1000000U);
    if (result != CY_RSLT_SUCCESS)
    {
        return (int32_t)result;
    }
    result = cyhal_timer_configure(&monotonic_timer, &config);
    if (result != CY_RSLT_SUCCESS)
    {
        return (int32_t)result;
    }
    return (int32_t)cyhal_timer_start(&monotonic_timer);
}

uint32_t timebase_now_us(void)
{
    return cyhal_timer_read(&monotonic_timer);
}

uint32_t timebase_now_ms(void)
{
    return timebase_now_us() / 1000U;
}

uint32_t timebase_elapsed_us(uint32_t start_us, uint32_t end_us)
{
    return end_us - start_us;
}

uint32_t timebase_elapsed_ms(uint32_t start_ms, uint32_t end_ms)
{
    return end_ms - start_ms;
}

bool timebase_reached_ms(uint32_t now_ms, uint32_t deadline_ms)
{
    return (int32_t)(now_ms - deadline_ms) >= 0;
}
