#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Portable helper used by firmware and host wrap-around tests. */
bool sample_window_copy_i16_ring(
    const int16_t *ring,
    size_t capacity,
    uint64_t total_written,
    uint64_t first_sample,
    size_t count,
    int16_t *destination);
