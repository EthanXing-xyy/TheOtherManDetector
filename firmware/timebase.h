#pragma once

#include <stdbool.h>
#include <stdint.h>

/* One shared 1 MHz monotonic counter timestamps Audio DMA endpoints and IMU
 * samples.  Unsigned differences are valid for intervals below 2^31 us. */
int32_t timebase_init(void);
uint32_t timebase_now_us(void);
uint32_t timebase_now_ms(void);
uint32_t timebase_elapsed_us(uint32_t start_us, uint32_t end_us);
uint32_t timebase_elapsed_ms(uint32_t start_ms, uint32_t end_ms);
bool timebase_reached_ms(uint32_t now_ms, uint32_t deadline_ms);
