# TheOtherManDetector | 隔壁老王检测器

### 中文版

> **你还在担心老婆出轨吗？**  
> **你还在担心长期出差在外，被隔壁老王偷家吗？**

为了保障异地夫妻的婚姻幸福生活，我们特此研发了 **TheOtherManDetector（隔壁老王检测器）**。

设想这样一个经典而令人不安的场景：

当你结束漫长的出差，终于回到家门口并敲响房门的那一刻，屋里的隔壁老王意识到大事不妙。正门显然已经不能走了，于是，他唯一合理的逃生方案就是：

**迅速拉开移门，从窗口跑路。**

问题来了。

正常人开门，和一个正在逃命的人开门，真的一样吗？

脚步的节奏、门体的振动、拉门的速度，以及整个过程中产生的声音，都可能暴露一些不太正常的行为模式。

于是，**TheOtherManDetector** 应运而生。

本项目利用同步采集的 **Audio + IMU 多模态信号**学习正常的移门交互模式。当系统检测到异常的脚步、异常的拉门动作或异常的声学与振动模式时，即可立即给出异常警报。

从技术角度来说，它实际上是一个运行于 **PSoC 6 MCU** 上的多模态 Edge AI 异常检测系统。

至于它究竟是在检测异常行为，还是在守护婚姻……

这取决于你怎么使用它。

### English Version

> **Still worried about your wife cheating on you?**  
> **Still worried that while you're away on a long business trip, *the other man* might be visiting your home?**

To protect the happiness and stability of long-distance marriages, we proudly present **TheOtherManDetector**.

Imagine a perfectly hypothetical scenario.

After a long business trip, you finally return home and knock on the door.

At that exact moment, *the other man* inside realizes that things have gone terribly wrong. The front door is no longer a viable escape route, leaving him with only one sensible option:

**pull open the sliding door and escape through the window.**

But here comes the engineering question:

Does a person opening a door normally behave the same way as someone opening it while desperately trying to escape?

Probably not.

Footstep patterns, door vibrations, opening speed, acoustic signatures, and motion dynamics may all change under unusual circumstances.

And that is exactly why **TheOtherManDetector** exists.

The system learns normal sliding-door interaction patterns from synchronized **Audio + IMU multimodal signals**. If unusual footsteps, abnormal door-opening motions, or suspicious acoustic and vibration patterns are detected, the system immediately raises an anomaly alert.

Technically speaking, this is a multimodal **Edge AI anomaly detection system deployed on a PSoC 6 MCU**.

**Whether it detects anomalous motion or protects your marriage is left as an exercise for the user.**

---

## Overview

A multimodal Audio-IMU Edge AI system for real-time behavioral anomaly detection on PSoC 6.

TheOtherManDetector analyzes events around a sliding door using a PDM
microphone and a six-axis IMU. It runs at the edge without a camera. The system
detects behavior that differs from its learned normal operating space; it does
not identify specific individuals and is not a gait-identity system.

The repository covers the complete path from synchronized recordings to an
embedded anomaly decision:

```text
Raw microphone + IMU recordings
              |
              v
   cleaning and onset alignment
              |
              v
  Audio and IMU preprocessing
              |
              v
 reconstruction representation learning
              |
              v
     encoder-only deployment model
              |
              v
          32-D embedding
              |
              v
 StandardScaler + K=2 prototypes +
 shared LedoitWolf Mahalanobis precision
              |
              v
       PT2 encoder export
              |
              v
    DEEPCRAFT Model Converter
              |
              v
       generated C/H model
              |
              v
 PSoC 6 synchronized real-time inference
```

Both the original raw dataset and cleaned/aligned dataset are included. The
deployment PT2, converter validation data, fitted anomaly parameters, generated
C/H, ModusToolbox descriptors, and deployment verification evidence are also
included, so building firmware does not require running the converter.

## Scenario Overview

<p align="center">
  <img src="docs/images/scenario_overview.png"
       alt="Sliding-door behavioral anomaly detection scenario"
       width="850">
</p>

## Hardware Setup

The firmware targets the Infineon CY8CKIT-062S2-AI PSoC 6 AI Evaluation Kit:

<p align="center">
  <img src="docs/images/hardware_setup.jpg"
       alt="PSoC 6 sliding-door hardware setup"
       width="750">
</p>

The board is mounted near the opening side of the lower sliding-door region.
The IMU captures door/track vibration and motion, while the PDM microphone
captures footsteps and door sounds.

- PSoC 6 Cortex-M4 application processor;
- onboard PDM microphone, captured as mono PCM16 at 16 kHz;
- BMI270 accelerometer and gyroscope, six axes at 100 Hz;
- camera-free sensing and local inference.

Sensor timing, range, gain, and the fixed event window are part of the deployed
signal contract and are not runtime tuning parameters.

## Dataset

The cleaned dataset has 164 accepted synchronized events:

| Split | Samples |
|---|---:|
| Normal Train | 123 |
| Normal Validation | 16 |
| Normal Test | 13 |
| Slight Anomaly | 6 |
| Fierce Anomaly | 6 |

Three labelled events were rejected because less than 2.5 seconds were
available before the detected onset. Their records and exact reasons remain in
metadata. See [dataset/README.md](dataset/README.md) and
[dataset/dataset_manifest.csv](dataset/dataset_manifest.csv).

## Signal processing

Each event spans `[-2.5, +4.7)` seconds around door-motion onset.

The neural audio input is not a raw waveform. Mono 16 kHz PCM is divided into
180 non-overlapping 40 ms frames of 640 samples. A periodic Hann window,
1024-point FFT power, 40 Slaney Mel bands over 0-7000 Hz,
`ln(mel + 1e-6)`, and global Normal Train per-band normalization produce
`[180, 40]` audio features. No per-sample amplitude normalization is used.

The IMU input is `[720, 6]` at 100 Hz in `ax, ay, az, gx, gy, gz` order, with
global Normal Train per-axis normalization. The network maps IMU timing onto
the synchronized internal representation.

The offline frequency study uses cumulative and transient spectral
energy from 22,140 Normal Train 40 ms spectra. Selecting 0-7000 Hz removes
approximately `0.416774%` aggregate energy above 7 kHz. Its code and analysis
tables/plots are in [`analysis/`](analysis).

## Model Architecture

The deployment network receives Audio as `180 × 40` log-Mel features and IMU
as a `720 × 6` six-axis sequence, and produces a 32-D embedding.

<p align="center">
  <img src="docs/images/model_architecture.png"
       alt="Multimodal Audio-IMU model architecture"
       width="900">
</p>

The reconstruction model combines:

- a 180-frame audio feature encoder;
- an IMU temporal encoder that maps 720 rows to the common 180-frame grid;
- fixed pooling to 90 synchronized global tokens;
- directed bipartite cross-modal feature attention;
- complete non-causal temporal attention;
- a one-layer unidirectional GRU;
- two temporal bottleneck convolutions;
- a 23 × 32 bottleneck and 32-D temporal-mean embedding.

Training uses a joint 180 × 64 reconstruction decoder. Deployment loads the
same encoder-side weights, excludes the decoder, and outputs only the 32-D
embedding. See [docs/architecture.md](docs/architecture.md).

## Anomaly detection

The neural embedding is followed by an external, fitted head:

```text
embedding -> StandardScaler -> distance to two normal prototypes
          -> minimum shared-precision Mahalanobis distance -> threshold
```

KMeans and LedoitWolf fitting run only on the PC. The MCU stores the fitted
mean, scale, two prototypes, and shared precision matrix. The normal embedding
space is represented by `K=2` prototypes fitted from Normal Train embeddings.

The base threshold is `5.692456722`, the Normal Validation maximum. Firmware
applies a separately named **Deployment Safety Factor** of `1.05`, giving a
deployment threshold of approximately `5.977079558`. The runtime decision rule
is:

```text
score > deployment threshold  => ANOMALY
score <= deployment threshold => NORMAL
```

## Model conversion

The static batch-one PT2 accepts float32 audio `[1, 180, 40]` and IMU
`[1, 720, 6]`, and returns `[1, 32]`. Deployment uses a fixed tanh-based GELU
approximation and nine state-linked ten-step GRU chunks to remain compatible
with the converter/runtime. Quantization is disabled.

[`converter/input`](converter/input) contains the deployment
`embedding_model.pt2` and `converter_validation.npz`, while
[`converter/project.byom`](converter/project.byom) stores the portable
DEEPCRAFT conversion configuration. The generated C/H model is already in
[`firmware/model`](firmware/model). See
[converter/README.md](converter/README.md) and
[docs/deployment.md](docs/deployment.md).

## Firmware deployment

The ModusToolbox application continuously acquires both sensors, recognizes a
door-opening event, captures the exact window, preprocesses both signals,
invokes `TMAI_compute(imu, audio, embedding)`, scores the embedding, and emits
`NORMAL` or `ANOMALY` over UART.

Dependencies are restored rather than vendored:

```powershell
cd firmware
make getlibs
make build CONFIG=Release
```

The Release build is required because the generated network requires `-O3` or
`-Ofast`. Detailed build, smoke-test, and programming instructions are in
[firmware/README.md](firmware/README.md).

## Results

Using the current deployment threshold, all 25 samples in the included
evaluation set produce the expected decision:

| Evaluation set | Correct decisions |
|---|---:|
| Normal Test | 13 / 13 |
| Slight Anomaly | 6 / 6 |
| Fierce Anomaly | 6 / 6 |

End-to-end raw-data replay on the target PSoC 6 achieved a maximum PC-to-MCU
anomaly-score error of `5.72e-6` with `0/25` decision mismatches.

The reported results describe the included dataset and are not intended as a
claim of universal field accuracy.

PT2 stored-reference parity had maximum absolute error `0.0`; the Python/C K=2
check had no decision mismatches. See
[results/evaluation_summary.csv](results/evaluation_summary.csv) and
[results/deployment_verification.md](results/deployment_verification.md).

## Repository structure

```text
dataset/       raw recordings, cleaned pairs, metadata, and manifest
analysis/      0-7000 Hz frequency-selection analysis
training/      preprocessing, modular models, training, fitting, export
converter/     PT2, validation NPZ, and portable DEEPCRAFT project
firmware/      buildable ModusToolbox application and generated C/H
tools/         reusable bit-preserving dataset cleaning utilities
docs/          architecture, deployment rationale, and limitations
results/       compact evaluation and verification evidence
```

## Quick start

Inspect and reproduce the PC-side pipeline from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r training/requirements.txt
python -m training.scripts.preprocess_dataset
python -m training.scripts.train
python -m training.scripts.fit_anomaly_detector
python -m training.scripts.export_model
```

These commands write reproducibility outputs under `training/outputs/` and do
not replace the checked-in deployment artifacts by default. To regenerate
the frequency study separately:

```powershell
python analysis/audio_frequency_analysis.py
```

## Known limitations

The dataset is relatively small and tied to a particular acoustic environment,
microphone path, door installation, and IMU mounting/orientation. Unseen normal
behavior can produce elevated scores, and background-noise or mechanical
domain shift can reduce separation. See
[docs/known_limitations.md](docs/known_limitations.md).

## Author

**Ethan_Xing**  

## License

Project code and documentation are released under the [MIT License](LICENSE).

The dataset under `dataset/` is released under
[CC0 1.0 Universal](dataset/LICENSE).

Third-party libraries, generated dependencies, and vendor components remain
subject to their respective licenses.
