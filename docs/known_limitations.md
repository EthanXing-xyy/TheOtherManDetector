# Known Limitations

- The dataset is relatively small, so the reported metrics describe the
  included evaluation set rather than population-level performance. It contains
  123 Normal Train, 16 Normal Validation, 13 Normal Test, six Slight Anomaly,
  and six Fierce Anomaly events.
- Environmental domain shift can change microphone spectra and IMU dynamics.
  Different rooms, door hardware, background noise, reverberation, and nearby
  activity may raise scores.
- Microphone characteristics, gain, placement, and background-noise levels may
  differ from the training installation.
- IMU mounting position, mechanical coupling, and axis orientation are part of
  the learned domain. Remounting or rotating the sensor can change the input
  distribution even when door behavior is normal.
- Unseen normal behaviors may lie outside the two fitted normal prototypes and
  produce elevated anomaly scores.
- Slight anomalies can overlap Normal behavior under environmental or sensor
  domain shift. Fierce anomalies are more strongly separated in the included
  evaluation set.
- The fixed 7.2-second event window assumes the configured door trigger and
  timing remain representative. It does not adapt window length to action
  speed.
