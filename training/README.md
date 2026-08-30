# Training and Export Pipeline

This directory contains the modular Python implementation for deterministic
preprocessing, reconstruction-based representation learning, encoder-only
deployment, K=2 anomaly-head fitting, and PT2 export. The included artifacts
are the deployment outputs; the commands below write reproducibility runs under
`training/outputs/` unless an explicit destination is supplied.

## Environment

Python 3.10 or newer is recommended. Install only the direct project
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r training/requirements.txt
```

PyTorch export and DEEPCRAFT compatibility can depend on the local PyTorch and
converter combination. The included PT2, portable DEEPCRAFT project, and
generated C/H define the deployment path.

## Data and preprocessing

[`config.yaml`](config.yaml) uses repository-relative defaults and points to
`dataset/processed`. Build normalized caches with the included Normal Train
statistics and Mel filterbank:

```powershell
python -m training.scripts.preprocess_dataset
```

To independently refit preprocessing statistics from the included 123 Normal
Train samples, write them to a separate directory:

```powershell
python -m training.scripts.preprocess_dataset `
  --recompute-statistics `
  --artifacts-dir training/outputs/preprocessing
```

The audio frontend is fixed:

1. read mono 16 kHz PCM16 and divide signed samples by 32768;
2. split 115,200 samples into 180 disjoint 640-sample frames;
3. apply a periodic Hann window;
4. zero-pad each frame to a 1024-point real FFT and calculate unnormalized
   squared-magnitude power;
5. apply 40 Slaney-scale, Slaney-area-normalized Mel filters over 0-7000 Hz;
6. calculate `ln(mel_energy + 1e-6)`;
7. apply global per-Mel Normal Train mean and standard deviation.

There is no per-sample amplitude normalization. IMU preprocessing loads
`[720, 6]` in `ax, ay, az, gx, gy, gz` order and applies global Normal Train
per-axis mean and standard deviation. It does not normalize individual samples.

## Reconstruction training

[`models/training_model.py`](models/training_model.py) contains the full
77,904-parameter training network, including its reconstruction-only decoder.
[`models/deployment_model.py`](models/deployment_model.py) builds the same
encoder side, loads those weights strictly, omits the decoder, and returns the
32-D temporal-mean embedding.

After preprocessing, run reconstruction training:

```powershell
python -m training.scripts.train
```

Only Normal Train is optimized. Normal Validation supplies the early-stopping
signal. Normal Test and anomalous samples are evaluation inputs only. The
included reconstruction checkpoint is
`checkpoints/reconstruction_model.pt`; reproducibility runs use functional
filenames under `outputs/checkpoints/`.

## Fit the anomaly detector

The anomaly head is fitted outside the neural network:

```powershell
python -m training.scripts.fit_anomaly_detector
```

The script evaluates the converter-compatible encoder one sample at a time and
fits a `StandardScaler` on the 123 Normal Train embeddings. KMeans uses `K=2`
to obtain two normal prototypes, and one shared LedoitWolf precision matrix is
fitted to the within-prototype residuals. The base threshold is the maximum
C-order float32 Mahalanobis score among the 16 Normal Validation embeddings.
The Deployment Safety Factor is applied only by the deployment configuration.

Fitted parameters are in `artifacts/mahalanobis_detector.npz`.

## Export PT2 and converter validation data

```powershell
python -m training.scripts.export_model
```

The exporter loads the selected reconstruction checkpoint, replaces GELU with
the fixed tanh approximation, preserves the state-linked 9 × 10-step GRU
structure, removes the decoder, and uses strict `torch.export` capture. It then
reloads the PT2 and creates a validation NPZ from all 16 Normal Validation
samples.

Generated reproduction outputs are written under `training/outputs/converter/`.
The deployment files are kept in [`converter/input`](../converter/input):

- `embedding_model.pt2`
- `converter_validation.npz`

The NPZ is converter validation data with stored references; it is neither
training data nor quantization calibration data. The portable converter
configuration is [`converter/project.byom`](../converter/project.byom).
