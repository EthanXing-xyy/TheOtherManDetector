# Dataset

This repository includes both the original synchronized recordings and the
fixed-window dataset derived from them. Audio PCM samples and numerical IMU
samples retain their source values, while public directories use semantic
split names.

## Contents

- `raw/` contains 12 acquisition sessions. Each session has a PDM microphone
  WAV recording, an IMU data file, human labels, and an acquisition-session
  descriptor.
- `processed/` contains aligned 7.2-second Audio/IMU pairs ready for feature
  preprocessing. Normal training sessions are flattened into one directory;
  the `session_NN_` prefix preserves their source identity.
- `dataset_manifest.csv` lists every raw recording, every accepted processed
  sample, and the three rejected cleaning records with file-derived sizes and
  sample counts.
- `processed/metadata/cleaning_metadata.csv` retains the alignment, integrity,
  onset, source-recording, and rejection details for all 167 labelled events.

## Sampling and alignment

Each accepted example is synchronized around the detected door-motion onset:

| Signal | Stored sample contract |
|---|---|
| Audio | mono PCM16, 16 kHz, 115,200 samples |
| IMU | 100 Hz, 720 rows, six axes |
| Accelerometer | `ax`, `ay`, `az`, m/s² |
| Gyroscope | `gx`, `gy`, `gz`, degrees/s |
| Window | `[-2.5, +4.7)` seconds, 7.2 seconds total |

The cleaning stage binds the onset to a source IMU row, copies the contiguous
720-row window, maps the same window into microphone time using the session
offset and actual WAV sample rate, and copies 115,200 contiguous PCM samples.
It does not interpolate, resample, pad, time-warp, filter, or normalize either
signal. Written audio is checked bit-for-bit and IMU axis text is checked
field-for-field against the source slice.

The onset score is only an alignment aid. It is calculated inside each human
label from a causal 100 ms normalized moving-RMS measure of six-axis IMU
differences. Saved IMU values remain raw.

## Splits

| Split | Label | Accepted samples | Role |
|---|---|---:|---|
| `normal_train` | Normal | 123 | representation training, normalization, and anomaly-head fitting |
| `normal_validation` | Normal | 16 | early stopping and statistical base-threshold construction |
| `normal_test` | Normal | 13 | included Normal evaluation set |
| `slight_anomaly` | Slight anomaly | 6 | included anomaly evaluation set |
| `fierce_anomaly` | Fierce anomaly | 6 | included anomaly evaluation set |

Normalization statistics are global population statistics (`ddof=0`) fitted
only on the 123 Normal Train samples. No per-sample normalization is used.

## Rejected records

Three labelled events are retained in metadata but have no processed signal
pair because the labelled interval did not provide the required 2.5 seconds
before the detected onset:

- `normal_train/session_03_sample_015`: 2.336999900 seconds available.
- `normal_train/session_05_sample_016`: 2.346999900 seconds available.
- `normal_validation/sample_017`: 2.158999900 seconds available.

The accepted processed count is therefore 164. Rejections never borrow signal
outside a human label and are not padded.

## Reproducing the cleaning stage

The bit-preserving alignment implementation is in
[`tools/clean_and_align_dataset.py`](../tools/clean_and_align_dataset.py). To
avoid overwriting the included processed dataset, its default output is under
`training/outputs/`:

```powershell
python -m tools.clean_and_align_dataset --input dataset/raw --output training/outputs/processed_dataset
```

## License

The dataset is released under the CC0 1.0 Universal dedication. See
[LICENSE](LICENSE) for details.
