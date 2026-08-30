# System Architecture

TheOtherManDetector is a camera-free, multimodal anomaly detector for behavior
around a sliding door. It learns a representation of normal Audio/IMU events
and compares each deployed embedding with two fitted normal prototypes. It
does not classify identity or recognize a specific individual.

## End-to-end data flow

```text
PDM microphone PCM + six-axis IMU
                 |
                 v
       synchronized 7.2 s event
                 |
        +--------+--------+
        |                 |
        v                 v
  180 x 40 log-Mel   720 x 6 IMU
        |                 |
        v                 v
   audio encoder       IMU encoder
        |                 |
        +------ 180-frame common grid
                         |
                  fixed pairwise pool
                         v
             90 synchronized tokens
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
   original tokens  cross-modal      temporal
                    attention        attention
        |                |                |
        +----------------+----------------+
                         v
                90 x 144 sequence
                         |
                  unidirectional GRU
                         v
                   90 x 64 sequence
                         |
              temporal bottleneck CNN
                         v
                   23 x 32 bottleneck
                         |
                   temporal mean
                         v
                   32-D embedding
                         |
             StandardScaler + K=2 shared-
             covariance Mahalanobis head
                         v
                anomaly score/decision
```

## External model inputs

The neural network does not take raw microphone PCM. Firmware and PC-side
preprocessing produce these normalized float32 tensors first:

- audio: `[batch, 180, 40]`, 40-band log-Mel features;
- IMU: `[batch, 720, 6]`, axes `ax, ay, az, gx, gy, gz`;
- deployment output: `[batch, 32]` embedding.

The exported PT2 uses a static batch size of one.

## Modality encoders

The audio branch applies `Conv1d(40, 32, kernel=1)` followed by
`Conv1d(32, 32, kernel=3, padding=1)`, with GELU-like activations. It preserves
the 180-frame time grid and returns 32 features per frame.

The IMU branch applies `Conv1d(6, 12, kernel=4, stride=4)`, mapping the 720
100 Hz rows onto the same 180-frame grid, then
`Conv1d(12, 16, kernel=3, padding=1)`. This timing mapping occurs inside the
network; the input remains the full 720 × 6 sequence.

Each branch is independently average-pooled with kernel and stride 2. The
global backbone therefore uses 90 tokens at an 80 ms spacing while the
external signal contract remains the 180-frame, 40 ms grid.

## Multimodal attention

The pooled audio and IMU features are concatenated into a 90 × 48 sequence.
Two attention paths consume that same synchronized representation:

- cross-modal feature attention is a directed bipartite graph. Each of the 32
  audio feature nodes attends only to the 16 IMU feature nodes, and each IMU
  node attends only to audio nodes;
- temporal attention is a complete, non-causal graph over all 90 timestamps.

Both use additive scores, LeakyReLU, source-wise softmax, weighted aggregation,
and sigmoid output. Concatenating the original, cross-modal, and temporal
representations produces 90 × 144 recurrent input.

## Recurrent and bottleneck representation

A one-layer, unidirectional GRU with hidden size 64 models the 90-token
sequence. Two stride-2 temporal convolutions map `64 -> 48 -> 32` channels and
`90 -> 45 -> 23` tokens. The deployment representation is the temporal mean of
the final 23 × 32 bottleneck, yielding 32 values.

The complete reconstruction model contains 77,904 learned parameters.

## Training-only reconstruction

Training resizes the 23-token bottleneck to 45 tokens without learned
parameters. The decoder then performs nearest-neighbor upsampling to 90 and
180 frames with convolutional refinement, producing a 180 × 64 target:

- 40 normalized audio features;
- 24 values formed by flattening each group of four normalized IMU rows.

The only optimized objective is unweighted mean squared error across this
joint target. Audio and IMU reconstruction losses are diagnostics. The decoder
is excluded from the deployment model and PT2.

## External anomaly head

The deployment embedding is standardized using mean and scale fitted on 123
Normal Train embeddings. KMeans provides two normal prototypes. A single
LedoitWolf precision matrix is fitted to standardized within-prototype
residuals. For prototype `k`:

```text
z = (embedding - mean) / scale
d_k = sqrt(max((z - prototype_k)^T precision (z - prototype_k), 0))
score = min(d_0, d_1)
```

The base decision boundary is the maximum score over the 16 Normal Validation
samples: `5.692456722`. Firmware applies the separately named Deployment Safety
Factor `1.05`, for a deployed decision boundary of approximately
`5.977079558`. All fitting happens on the PC; the MCU stores only fitted
parameters.
