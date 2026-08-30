# Deployment Verification

## On-Device End-to-End Verification

The complete deployment pipeline was verified on the target PSoC 6:

```text
Raw Audio + Raw IMU
        -> MCU preprocessing
        -> generated neural model
        -> 32-D embedding
        -> K=2 Mahalanobis anomaly score
        -> final decision
```

| Check | Result |
| --- | ---: |
| Raw evaluation samples replayed on PSoC 6 | 25 |
| Maximum IMU preprocessing error | 0 |
| Maximum Audio preprocessing error | `2.20537186e-6` |
| Maximum embedding error | `1.43051147e-6` |
| Maximum anomaly-score error | `5.72204590e-6` |
| PC-to-MCU decision mismatches | 0 / 25 |

The deployment-threshold decisions are:

| Evaluation set | Correct decisions |
| --- | ---: |
| Normal Test | 13 / 13 |
| Slight Anomaly | 6 / 6 |
| Fierce Anomaly | 6 / 6 |

Total: **25 / 25 expected decisions** in the included evaluation set.

All 25 samples in the included evaluation set produce the expected decision
under the current deployment threshold. These results describe the included
evaluation set and are not a claim of universal or population-level accuracy.

## Supporting Technical Checks

| Check | Result |
|---|---:|
| Processed Audio/IMU sample integrity checks | 164 |
| Checkpoint parameters | 77,904 |
| PT2 parameter tensors | 18 encoder-side tensors |
| PT2 validation output | `mean`, float32 `[16, 32]` |
| Reloaded PT2 versus stored NPZ references | maximum absolute error `0.0` |
| Native versus nine-chunk GRU embedding | maximum absolute error `7.152557e-7` |
| Exact GELU versus tanh approximation | maximum absolute error `5.738735e-4` |
| Exact GELU versus tanh minimum cosine similarity | `0.9999999936` |
| Python versus compiled C K=2 decisions | 0 mismatches |
| Python versus compiled C nearest prototype | 0 mismatches |

The exported graph has static batch-one inputs: audio `[1, 180, 40]` and IMU
`[1, 720, 6]`. It contains nine state-linked, ten-step calls to one shared GRU
and a maximum CONCAT arity of nine. The training-only decoder is absent.

The Release firmware and validation-smoke configurations compile and link for
`APP_CY8CKIT-062S2-AI`. The deployment pipeline uses the generated DEEPCRAFT C
model for PSoC 6 inference. See
[evaluation_summary.csv](evaluation_summary.csv).
