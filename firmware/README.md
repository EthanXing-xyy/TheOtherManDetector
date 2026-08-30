# PSoC 6 Firmware

`firmware/` is a complete ModusToolbox application root for the
`APP_CY8CKIT-062S2-AI` target (CY8CKIT-062S2-AI, PSoC 6 Cortex-M4). It performs
continuous microphone and BMI270 acquisition, door-event windowing, signal
preprocessing, neural embedding inference, and Mahalanobis anomaly scoring.

## Requirements

- Infineon ModusToolbox with the GCC ARM toolchain
- CY8CKIT-062S2-AI for real sensor execution
- KitProg3 USB connection for programming/debugging
- A serial terminal at 115200 baud, 8N1, for runtime messages

The checked-in `Makefile`, `deps/*.mtb`, `libs/*.mtb`, lockfile, and BSP files
are the dependency descriptors. The large generated `mtb_shared/` dependency
tree is intentionally not committed.

## Restore dependencies and build

From `firmware/`:

```powershell
make getlibs
make build CONFIG=Release
```

`CY_GETLIBS_SHARED_PATH=../` places the restored shared tree at repository-root
`mtb_shared/`. That directory and firmware build output are ignored by Git.

Always use the Release configuration for the generated network. The generated
source explicitly requires `-O3` or `-Ofast`; this Makefile adds `-O3` for
Release builds.

To build the optional stored-reference smoke configuration:

```powershell
make build CONFIG=Release ENABLE_VALIDATION_SMOKE_TEST=1 REAL_SENSOR_MODE=0
```

## Program and debug

With the kit attached through KitProg3, program a Release build from the
ModusToolbox IDE Quick Panel or run:

```powershell
make program CONFIG=Release
```

The deployment pipeline targets the CY8CKIT-062S2-AI and uses the generated
DEEPCRAFT C model for on-device embedding inference.

## Runtime signal path

[`microphone_capture.c`](microphone_capture.c) continuously captures the PDM
microphone as mono PCM16 at 16 kHz. The BMI270 supplies three accelerometer and
three gyroscope axes at 100 Hz. A door opening event latches a synchronized
`[-2.5, +4.7)` second window:

- audio: 115,200 PCM samples;
- IMU: 720 × 6 samples.

Audio preprocessing creates 180 × 40 normalized log-Mel features using 640
samples per frame, a periodic Hann window, 1024-point FFT power, 0-7000 Hz,
Slaney Mel filters, `ln(mel + 1e-6)`, and global Normal Train per-band
statistics. IMU preprocessing applies global Normal Train per-axis statistics.
Neither path performs per-sample normalization.

[`model/embedding_model.c`](model/embedding_model.c) exposes the generated API:

```c
TMAI_compute(imu, audio, embedding);
```

The C API orders IMU before audio and returns a 32-D embedding. The generated
model reserves 451,584 bytes of state RAM, 8 bytes of buffer RAM, and 603,148
bytes of read-only model data according to its preserved generator header.

## Anomaly score

[`model/mahalanobis_detector.c`](model/mahalanobis_detector.c) applies the
fitted external head:

1. standardize the 32-D embedding with stored Normal Train mean/scale;
2. calculate Mahalanobis distance to each of two stored normal prototypes;
3. use their shared stored LedoitWolf precision matrix;
4. set `score = min(distance_0, distance_1)`;
5. report an anomaly when the score exceeds the deployment threshold.

KMeans and covariance fitting never run on the MCU.

The decision constants remain separate and visible in `app_config.h`:

```c
#define BASE_ANOMALY_THRESHOLD (5.692456722f)
#define ANOMALY_THRESHOLD_SAFETY_FACTOR (1.05f)
#define DEPLOYMENT_ANOMALY_THRESHOLD \
    (BASE_ANOMALY_THRESHOLD * ANOMALY_THRESHOLD_SAFETY_FACTOR)
```

The base threshold is the Normal Validation maximum. The **Deployment Safety
Factor** provides 5% additional tolerance for installation, sensor, and
environmental variation, producing approximately `5.977079558`. It changes
only the decision boundary, not the embedding or Mahalanobis mathematics.

The runtime decision rule is:

```text
score > deployment threshold  => ANOMALY
score <= deployment threshold => NORMAL
```

Runtime output reports the score, threshold, and `NORMAL` or `ANOMALY`. The
system detects behavior outside its learned normal space; it does not identify
people.
