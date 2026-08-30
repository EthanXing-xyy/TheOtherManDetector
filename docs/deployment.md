# Deployment Design

The deployment path keeps the trained encoder-side mathematics and learned
weights while expressing the graph in operations supported by DEEPCRAFT Model
Converter and the PSoC 6 TensorFlow Lite Micro runtime.

## Signal contract

Firmware continuously captures a 16 kHz mono PDM microphone stream and a
100 Hz six-axis BMI270 stream. A confirmed door-opening event selects a fixed
7.2-second window with 2.5 seconds before and 4.7 seconds after the event time.

Raw PCM is transformed before inference. Each of 180 disjoint 40 ms frames
contains 640 samples. The implementation applies a periodic Hann window,
1024-point real FFT, squared-magnitude power, a 40-band Slaney Mel filterbank
over 0-7000 Hz, `ln(mel + 1e-6)`, and global Normal Train per-band
normalization. The FFT power is not amplitude-normalized and there is no
sample-dependent amplitude normalization.

The IMU tensor remains 720 × 6 and receives global Normal Train per-axis
normalization. The first model convolution maps each four-row IMU block to the
common 180-frame representation.

## Tanh-based GELU

Native GELU is not supported by the target conversion/runtime path. Deployment
uses the fixed approximation:

```text
0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))
```

This keeps the graph converter-compatible while preserving GELU-like behavior.
The formula is used in both modality encoders and both bottleneck activations;
its mathematics must not be changed independently of the deployment PT2 and C
model. Across all 164 cleaned samples, the exact-GELU versus approximation
embedding maximum absolute difference was `0.0005738735`, with minimum cosine
similarity `0.9999999936`.

## Chunked recurrent graph

The runtime supports at most ten inputs on a converted CONCAT node. The
90-token recurrent sequence is processed as nine state-linked chunks of ten
steps through one shared GRU. The final hidden state from each chunk initializes
the next chunk, so recurrence remains continuous.

The nine outputs are concatenated in temporal order in the exported graph. The
maximum CONCAT arity is nine, within the runtime limit. The explicit chunked
structure keeps converted CONCAT nodes within the runtime-supported input-count
limit. Native versus chunked embedding maximum absolute difference was
`7.152557e-7` across the dataset.

## PT2 and DEEPCRAFT

The deployment PT2 has static batch-one inputs:

- `audio`: float32 `[1, 180, 40]`;
- `imu`: float32 `[1, 720, 6]`;
- output: float32 `[1, 32]`.

The converter validation NPZ contains 16 Normal Validation samples and their
reloaded-PT2 references under keys `audio`, `imu`, and `mean`. Quantization is
disabled; this data validates conversion and is not calibration data.

The generated implementation uses the TFLM runtime and exposes
`TMAI_compute(imu, audio, data_out)`. The generated C API orders IMU before
audio even though the Python model signature is `(audio, imu)`. The preserved
generator header reports:

- 8 bytes buffer RAM;
- 451,584 bytes state RAM;
- 603,148 bytes read-only model data;
- DEEPCRAFT/ImagiNet Compiler 5.11.5765.0.

The generated source requires compiler optimization `-O3` or `-Ofast`.
`make build CONFIG=Release` supplies `-O3`.

See [`converter/README.md`](../converter/README.md) for the converter
settings, portable [`project.byom`](../converter/project.byom), and exact
project-based CLI command.

## Anomaly decision boundary

The neural graph returns only the 32-D embedding. Firmware then uses stored
StandardScaler parameters, two KMeans normal prototypes, and a shared
LedoitWolf precision matrix. The score is the smaller of the two Mahalanobis
distances.

The firmware constants remain separated:

- base anomaly threshold: `5.692456722`, selected as the Normal Validation
  maximum;
- Deployment Safety Factor: `1.05`;
- deployment threshold: base × factor, approximately `5.977079558`.

The safety factor changes only the binary decision boundary. It does not alter
model inputs, weights, embeddings, prototypes, precision, or scores.

```text
score > deployment threshold  => ANOMALY
score <= deployment threshold => NORMAL
```

## Verification scope

The stored PT2 reproduces every reference embedding in
`converter_validation.npz` exactly. The Python/C anomaly-head parity check has
no decision or nearest-prototype mismatches. The Release firmware and optional
validation-smoke configuration compile and link for `APP_CY8CKIT-062S2-AI`.
The deployment pipeline uses the generated DEEPCRAFT C model for PSoC 6
inference.
