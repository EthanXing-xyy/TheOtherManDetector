#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "sensor_imu.h"

typedef enum
{
    DOOR_STATE_WAIT_OPEN = 0,
    DOOR_STATE_CAPTURE_OPEN_WINDOW,
    DOOR_STATE_WAIT_DOOR_STABLE,
    DOOR_STATE_WAIT_CLOSE,
    DOOR_STATE_CLOSE_COOLDOWN,
} door_state_t;

typedef enum
{
    DOOR_UPDATE_NONE = 0,
    DOOR_UPDATE_OPEN_CONFIRMED,
    DOOR_UPDATE_IMU_WINDOW_COMPLETE,
    DOOR_UPDATE_OPEN_SETTLED,
    DOOR_UPDATE_CLOSE_DETECTED,
    DOOR_UPDATE_REARMED,
} door_update_kind_t;

typedef struct
{
    door_update_kind_t kind;
    door_state_t state;
    uint32_t event_id;
    uint32_t trigger_us;
    uint32_t trigger_ms;
    uint32_t trigger_sequence;
    float delta_ay;
    float acc_var;
    float gyro_var;
} door_update_t;

void event_detector_init(void);
door_state_t event_detector_state(void);
bool event_detector_ready_for_open(void);
door_update_t event_detector_accept(const imu_sample_t *sample, bool audio_history_ready);
bool event_detector_begin_wait_open_settle(void);
uint32_t event_detector_event_id(void);
uint32_t event_detector_t0_us(void);
uint32_t event_detector_t0_ms(void);
uint32_t event_detector_t0_sequence(void);
