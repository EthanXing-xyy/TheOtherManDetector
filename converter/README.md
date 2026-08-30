# DEEPCRAFT Model Conversion

Normal users do not need DEEPCRAFT Model Converter. The generated C/H
implementation is already included in [`firmware/model`](../firmware/model)
and is built directly with the firmware application.

Advanced users can regenerate the implementation from the deployment PT2,
validation data with stored reference outputs, and the portable DEEPCRAFT
converter project in this directory.

## Inputs

### `input/embedding_model.pt2`

This is the static, batch-one, float32 PyTorch ExportedProgram for the
encoder-side deployment network. Its public signature is:

| Name | Direction | Shape | Type | Meaning |
|---|---|---:|---|---|
| `audio` | input | `[1, 180, 40]` | float32 | normalized 40-band log-Mel features |
| `imu` | input | `[1, 720, 6]` | float32 | normalized six-axis IMU sequence |
| output | output | `[1, 32]` | float32 | temporal-mean embedding |

The PT2 does not accept raw microphone PCM. Runtime audio preprocessing occurs
before neural inference. The training-only reconstruction decoder and the
external Mahalanobis detector are not part of this graph.

### `input/converter_validation.npz`

The archive contains exactly these arrays:

| Key | Shape | Type |
|---|---:|---|
| `audio` | `[16, 180, 40]` | float32 |
| `imu` | `[16, 720, 6]` | float32 |
| `mean` | `[16, 32]` | float32 |

`mean` contains reference outputs from the reloaded deployment PT2, evaluated
one sample at a time. This archive is validation data, not model-training data
and not quantization calibration data.

### `project.byom`

This is the saved DEEPCRAFT Model Converter project for the deployed model. It
uses project-relative paths to `input/embedding_model.pt2` and
`input/converter_validation.npz`, writes generated material under the ignored
`converter/output/` directory, and preserves the converter settings
listed below.

## Converter configuration

The generated implementation records DEEPCRAFT Model Converter
`5.11.5765.0` and ImagiNet Compiler `5.11.5765.0`. The deployment conversion
configuration is:

- Target Device: PSoC 6
- Model is RNN: enabled
- Interpreterless: disabled
- Network Quantization: disabled
- Validation source: NPZ
- Maximum validation samples: 16
- Validation mode: Use Provided Reference

The generated runtime is TensorFlow Lite Micro and keeps float32 inputs,
weights, state, and output.

## CLI reference

The CLI provides help for the relevant conversion commands:

```powershell
dcmc --help
dcmc convert --help
dcmc convert-project --help
dcmc pytorch --help
```

Version `5.11.5765.0` supports `convert-project` and requires the project with
`-p`/`--project`. From the repository root, the project command is:

```powershell
dcmc convert-project -p converter/project.byom -o converter/output
```

PyTorch ExportedProgram conversion requires either an authenticated DEEPCRAFT
cloud connection or a configured local converter container. Inspect the active
configuration with:

```powershell
dcmc auth --status
dcmc pytorch --status
dcmc pytorch --test
```

## Regenerating the model

1. Install DEEPCRAFT Model Converter `5.11.5765.0` or a compatible release and
   configure its PyTorch conversion service.
2. Run the `convert-project` command above from the repository root.
3. Review the validation report and compare the generated API with the
   included firmware API before replacing files.
   The required call is equivalent to
   `TMAI_compute(const float *imu, const float *audio, float *data_out)`; note
   that the generated C API orders IMU before audio.
4. Place reviewed outputs at `firmware/model/embedding_model.c` and
   `firmware/model/embedding_model.h`, updating only the public header include
   if the converter emits its own filename.

The generated source requires `-O3` or `-Ofast`. The firmware Release build
adds `-O3`.
