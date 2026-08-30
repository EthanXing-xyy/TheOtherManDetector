#include "sample_window.h"

#include <string.h>

bool sample_window_copy_i16_ring(
    const int16_t *ring,
    size_t capacity,
    uint64_t total_written,
    uint64_t first_sample,
    size_t count,
    int16_t *destination)
{
    uint64_t oldest;
    uint64_t end;
    size_t first_index;
    size_t first_count;

    if ((ring == NULL) || (destination == NULL) || (capacity == 0U) ||
        ((capacity & (capacity - 1U)) != 0U) || (count > capacity))
    {
        return false;
    }
    if (first_sample > UINT64_MAX - (uint64_t)count)
    {
        return false;
    }
    end = first_sample + (uint64_t)count;
    oldest = (total_written > (uint64_t)capacity) ?
        total_written - (uint64_t)capacity : 0U;
    if ((first_sample < oldest) || (end > total_written))
    {
        return false;
    }

    first_index = (size_t)(first_sample & (uint64_t)(capacity - 1U));
    first_count = capacity - first_index;
    if (first_count > count)
    {
        first_count = count;
    }
    memcpy(destination, &ring[first_index], first_count * sizeof(*destination));
    if (first_count < count)
    {
        memcpy(
            destination + first_count,
            ring,
            (count - first_count) * sizeof(*destination));
    }
    return true;
}
