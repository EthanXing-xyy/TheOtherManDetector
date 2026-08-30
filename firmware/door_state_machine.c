#include "door_state_machine.h"

#include <math.h>
#include <string.h>

#include "app_config.h"

typedef struct
{
    float values[IMU_BASELINE_FRAMES][IMU_ACCEL_AXES];
    float sum[IMU_ACCEL_AXES];
    size_t head;
    size_t count;
} acceleration_window_t;

typedef struct
{
    float values[OPEN_SETTLE_WINDOW_FRAMES][IMU_CHANNELS];
    size_t head;
    size_t count;
} settle_window_t;

static door_state_t door_state;
static acceleration_window_t baseline_window;
static settle_window_t settle_window;
static float frozen_baseline[IMU_ACCEL_AXES];
static uint32_t open_candidate_frames;
static uint32_t close_candidate_frames;
static uint32_t current_event_id;
static uint32_t current_t0_us;
static uint32_t current_t0_ms;
static uint32_t current_t0_sequence;
static float current_open_delta_ay;
static uint32_t close_candidate_ms;
static uint32_t close_candidate_us;
static uint32_t close_candidate_sequence;
static float close_candidate_delta_ay;
static bool imu_window_complete_reported;

static door_update_t make_update(
    door_update_kind_t kind,
    uint32_t trigger_ms,
    uint32_t trigger_sequence,
    float delta_ay)
{
    door_update_t update = {
        .kind = kind,
        .state = door_state,
        .event_id = current_event_id,
        .trigger_us = current_t0_us,
        .trigger_ms = trigger_ms,
        .trigger_sequence = trigger_sequence,
        .delta_ay = delta_ay,
        .acc_var = 0.0f,
        .gyro_var = 0.0f,
    };
    return update;
}

static door_update_t no_update(void)
{
    return make_update(DOOR_UPDATE_NONE, 0U, 0U, 0.0f);
}

static void acceleration_window_reset(acceleration_window_t *window)
{
    memset(window, 0, sizeof(*window));
}

static void acceleration_window_push(
    acceleration_window_t *window,
    const float acceleration[IMU_ACCEL_AXES])
{
    if (window->count == IMU_BASELINE_FRAMES)
    {
        for (size_t axis = 0U; axis < IMU_ACCEL_AXES; ++axis)
        {
            window->sum[axis] -= window->values[window->head][axis];
        }
    }
    else
    {
        ++window->count;
    }

    for (size_t axis = 0U; axis < IMU_ACCEL_AXES; ++axis)
    {
        window->values[window->head][axis] = acceleration[axis];
        window->sum[axis] += acceleration[axis];
    }
    window->head = (window->head + 1U) % IMU_BASELINE_FRAMES;
}

static void acceleration_window_mean(
    const acceleration_window_t *window,
    float mean[IMU_ACCEL_AXES])
{
    for (size_t axis = 0U; axis < IMU_ACCEL_AXES; ++axis)
    {
        mean[axis] = window->sum[axis] / (float)window->count;
    }
}

static void settle_window_reset(settle_window_t *window)
{
    memset(window, 0, sizeof(*window));
}

static void settle_window_push(
    settle_window_t *window,
    const float values[IMU_CHANNELS])
{
    memcpy(window->values[window->head], values, sizeof(window->values[0]));
    window->head = (window->head + 1U) % OPEN_SETTLE_WINDOW_FRAMES;
    if (window->count < OPEN_SETTLE_WINDOW_FRAMES)
    {
        ++window->count;
    }
}

static void settle_window_variances(
    const settle_window_t *window,
    float *acc_var,
    float *gyro_var)
{
    float means[IMU_CHANNELS] = {0.0f};
    float variances[IMU_CHANNELS] = {0.0f};

    for (size_t frame = 0U; frame < OPEN_SETTLE_WINDOW_FRAMES; ++frame)
    {
        for (size_t axis = 0U; axis < IMU_CHANNELS; ++axis)
        {
            means[axis] += window->values[frame][axis];
        }
    }
    for (size_t axis = 0U; axis < IMU_CHANNELS; ++axis)
    {
        means[axis] /= (float)OPEN_SETTLE_WINDOW_FRAMES;
    }

    for (size_t frame = 0U; frame < OPEN_SETTLE_WINDOW_FRAMES; ++frame)
    {
        for (size_t axis = 0U; axis < IMU_CHANNELS; ++axis)
        {
            float difference = window->values[frame][axis] - means[axis];
            variances[axis] += difference * difference;
        }
    }
    for (size_t axis = 0U; axis < IMU_CHANNELS; ++axis)
    {
        variances[axis] /= (float)OPEN_SETTLE_WINDOW_FRAMES;
        if (variances[axis] < 0.0f)
        {
            variances[axis] = 0.0f;
        }
    }

    *acc_var = variances[0] + variances[1] + variances[2];
    *gyro_var = variances[3] + variances[4] + variances[5];
}

static void rebuild_acceleration_baseline_from_settle(
    const settle_window_t *window)
{
    acceleration_window_reset(&baseline_window);
    for (size_t offset = 0U; offset < OPEN_SETTLE_WINDOW_FRAMES; ++offset)
    {
        /* window->head is the oldest element after a full rolling window. */
        size_t frame = (window->head + offset) % OPEN_SETTLE_WINDOW_FRAMES;
        acceleration_window_push(&baseline_window, window->values[frame]);
    }
}

static float acceleration_motion(
    const float acceleration[IMU_ACCEL_AXES],
    const float baseline[IMU_ACCEL_AXES])
{
    float squared_norm = 0.0f;
    for (size_t axis = 0U; axis < IMU_ACCEL_AXES; ++axis)
    {
        float difference = acceleration[axis] - baseline[axis];
        squared_norm += difference * difference;
    }
    return sqrtf(squared_norm);
}

static float open_direction_metric(float delta_axis)
{
    return OPEN_DIRECTION_SIGN * delta_axis;
}

static float close_direction_metric(float delta_axis)
{
    return -OPEN_DIRECTION_SIGN * delta_axis;
}

void event_detector_init(void)
{
    door_state = DOOR_STATE_WAIT_OPEN;
    acceleration_window_reset(&baseline_window);
    settle_window_reset(&settle_window);
    memset(frozen_baseline, 0, sizeof(frozen_baseline));
    open_candidate_frames = 0U;
    close_candidate_frames = 0U;
    current_event_id = 0U;
    current_t0_us = 0U;
    current_t0_ms = 0U;
    current_t0_sequence = 0U;
    current_open_delta_ay = 0.0f;
    close_candidate_ms = 0U;
    close_candidate_us = 0U;
    close_candidate_sequence = 0U;
    close_candidate_delta_ay = 0.0f;
    imu_window_complete_reported = false;
}

door_state_t event_detector_state(void)
{
    return door_state;
}

bool event_detector_ready_for_open(void)
{
    return (door_state == DOOR_STATE_WAIT_OPEN) &&
        (baseline_window.count == IMU_BASELINE_FRAMES) &&
        (open_candidate_frames == 0U);
}

door_update_t event_detector_accept(
    const imu_sample_t *sample,
    bool audio_history_ready)
{
    float baseline[IMU_ACCEL_AXES];
    float delta_ay;
    float motion;

    if (sample == NULL)
    {
        return no_update();
    }

    switch (door_state)
    {
        case DOOR_STATE_WAIT_OPEN:
            if (baseline_window.count < IMU_BASELINE_FRAMES)
            {
                acceleration_window_push(&baseline_window, sample->values);
                return no_update();
            }

            if (open_candidate_frames != 0U)
            {
                delta_ay = sample->values[DOOR_DIRECTION_AXIS] -
                    frozen_baseline[DOOR_DIRECTION_AXIS];
                if (!audio_history_ready ||
                    (open_direction_metric(delta_ay) <=
                     OPEN_DIRECTION_MAGNITUDE_MPS2))
                {
                    open_candidate_frames = 0U;
                    return no_update();
                }

                ++open_candidate_frames;
                if (open_candidate_frames >= OPEN_CONSECUTIVE_FRAMES)
                {
                    ++current_event_id;
                    open_candidate_frames = 0U;
                    imu_window_complete_reported = false;
                    door_state = DOOR_STATE_CAPTURE_OPEN_WINDOW;
                    return make_update(
                        DOOR_UPDATE_OPEN_CONFIRMED,
                        current_t0_ms,
                        current_t0_sequence,
                        current_open_delta_ay);
                }
                return no_update();
            }

            acceleration_window_mean(&baseline_window, baseline);
            delta_ay = sample->values[DOOR_DIRECTION_AXIS] -
                baseline[DOOR_DIRECTION_AXIS];
            motion = acceleration_motion(sample->values, baseline);
            if (audio_history_ready &&
                (open_direction_metric(delta_ay) >
                 OPEN_DIRECTION_MAGNITUDE_MPS2))
            {
                memcpy(frozen_baseline, baseline, sizeof(frozen_baseline));
                current_t0_us = sample->timestamp_us;
                current_t0_ms = sample->timestamp_ms;
                current_t0_sequence = sample->sequence;
                current_open_delta_ay = delta_ay;
                open_candidate_frames = 1U;
            }
            else if (motion <= MOTION_THRESHOLD_MPS2)
            {
                acceleration_window_push(&baseline_window, sample->values);
            }
            return no_update();

        case DOOR_STATE_CAPTURE_OPEN_WINDOW:
            if (!imu_window_complete_reported &&
                ((uint32_t)(sample->sequence - current_t0_sequence) >=
                 (IMU_POSTTRIGGER_FRAMES - 1U)))
            {
                imu_window_complete_reported = true;
                return make_update(
                    DOOR_UPDATE_IMU_WINDOW_COMPLETE,
                    current_t0_ms,
                    current_t0_sequence,
                    current_open_delta_ay);
            }
            return no_update();

        case DOOR_STATE_WAIT_DOOR_STABLE:
        {
            float acc_var;
            float gyro_var;

            settle_window_push(&settle_window, sample->values);
            if (settle_window.count < OPEN_SETTLE_WINDOW_FRAMES)
            {
                return no_update();
            }

            settle_window_variances(&settle_window, &acc_var, &gyro_var);
            if ((acc_var < OPEN_SETTLE_ACC_VAR_THRESHOLD) &&
                (gyro_var < OPEN_SETTLE_GYRO_VAR_THRESHOLD))
            {
                door_update_t update;

                rebuild_acceleration_baseline_from_settle(&settle_window);
                settle_window_reset(&settle_window);
                close_candidate_frames = 0U;
                door_state = DOOR_STATE_WAIT_CLOSE;
                update = make_update(
                    DOOR_UPDATE_OPEN_SETTLED,
                    sample->timestamp_ms,
                    sample->sequence,
                    0.0f);
                update.acc_var = acc_var;
                update.gyro_var = gyro_var;
                return update;
            }
            return no_update();
        }

        case DOOR_STATE_WAIT_CLOSE:
            if (baseline_window.count < IMU_BASELINE_FRAMES)
            {
                acceleration_window_push(&baseline_window, sample->values);
                return no_update();
            }

            if (close_candidate_frames != 0U)
            {
                delta_ay = sample->values[DOOR_DIRECTION_AXIS] -
                    frozen_baseline[DOOR_DIRECTION_AXIS];
                if (close_direction_metric(delta_ay) <=
                    CLOSE_DIRECTION_MAGNITUDE_MPS2)
                {
                    close_candidate_frames = 0U;
                    return no_update();
                }

                ++close_candidate_frames;
                if (close_candidate_frames >= CLOSE_CONSECUTIVE_FRAMES)
                {
                    close_candidate_frames = 0U;
                    /* The WAIT_CLOSE baseline describes the open pose.  It
                     * must never be reused after the door closes. */
                    acceleration_window_reset(&baseline_window);
                    settle_window_reset(&settle_window);
                    door_state = DOOR_STATE_CLOSE_COOLDOWN;
                    return make_update(
                        DOOR_UPDATE_CLOSE_DETECTED,
                        close_candidate_ms,
                        close_candidate_sequence,
                        close_candidate_delta_ay);
                }
                return no_update();
            }

            acceleration_window_mean(&baseline_window, baseline);
            delta_ay = sample->values[DOOR_DIRECTION_AXIS] -
                baseline[DOOR_DIRECTION_AXIS];
            motion = acceleration_motion(sample->values, baseline);
            if (close_direction_metric(delta_ay) >
                CLOSE_DIRECTION_MAGNITUDE_MPS2)
            {
                memcpy(frozen_baseline, baseline, sizeof(frozen_baseline));
                close_candidate_us = sample->timestamp_us;
                close_candidate_ms = sample->timestamp_ms;
                close_candidate_sequence = sample->sequence;
                close_candidate_delta_ay = delta_ay;
                close_candidate_frames = 1U;
            }
            else if (motion <= MOTION_THRESHOLD_MPS2)
            {
                acceleration_window_push(&baseline_window, sample->values);
            }
            return no_update();

        case DOOR_STATE_CLOSE_COOLDOWN:
        {
            float acc_var;
            float gyro_var;

            /* Continuously retain the most recent 500 ms while the close
             * motion and mechanical ringing decay. */
            settle_window_push(&settle_window, sample->values);
            if (((uint32_t)(sample->timestamp_us - close_candidate_us) <
                 DOOR_REARM_DELAY_US) ||
                (settle_window.count < OPEN_SETTLE_WINDOW_FRAMES))
            {
                return no_update();
            }

            settle_window_variances(&settle_window, &acc_var, &gyro_var);
            if ((acc_var < OPEN_SETTLE_ACC_VAR_THRESHOLD) &&
                (gyro_var < OPEN_SETTLE_GYRO_VAR_THRESHOLD))
            {
                door_update_t update;

                rebuild_acceleration_baseline_from_settle(&settle_window);
                settle_window_reset(&settle_window);
                open_candidate_frames = 0U;
                close_candidate_frames = 0U;
                door_state = DOOR_STATE_WAIT_OPEN;
                update = make_update(
                    DOOR_UPDATE_REARMED,
                    sample->timestamp_ms,
                    sample->sequence,
                    0.0f);
                update.acc_var = acc_var;
                update.gyro_var = gyro_var;
                return update;
            }
            return no_update();
        }

        default:
            return no_update();
    }
}

bool event_detector_begin_wait_open_settle(void)
{
    if (door_state != DOOR_STATE_CAPTURE_OPEN_WINDOW)
    {
        return false;
    }

    acceleration_window_reset(&baseline_window);
    settle_window_reset(&settle_window);
    open_candidate_frames = 0U;
    close_candidate_frames = 0U;
    door_state = DOOR_STATE_WAIT_DOOR_STABLE;
    return true;
}

uint32_t event_detector_event_id(void)
{
    return current_event_id;
}

uint32_t event_detector_t0_us(void)
{
    return current_t0_us;
}

uint32_t event_detector_t0_ms(void)
{
    return current_t0_ms;
}

uint32_t event_detector_t0_sequence(void)
{
    return current_t0_sequence;
}
