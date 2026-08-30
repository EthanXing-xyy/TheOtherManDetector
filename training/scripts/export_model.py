#!/usr/bin/env python
"""Export the converter-compatible encoder and its validation NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from training.data import CachedMultimodalDataset, load_metadata, rows_for_split
from training.models.deployment_model import load_checkpoint_encoder
from training.utils import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "training" / "config.yaml"
DEFAULT_CHECKPOINT = REPO_ROOT / "training" / "checkpoints" / "reconstruction_model.pt"


def parse_args() -> argparse.Namespace:
    output_root = REPO_ROOT / "training" / "outputs" / "converter"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-model", type=Path, default=output_root / "embedding_model.pt2")
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=output_root / "converter_validation.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    metadata = load_metadata(config)
    cache_root = args.cache_dir.resolve() if args.cache_dir else Path(config["paths"]["cache_dir"])
    validation = CachedMultimodalDataset(
        cache_root, rows_for_split(config, metadata, "validation")
    )
    if len(validation) != 16:
        raise AssertionError("Converter validation requires all 16 Normal Validation samples")
    model = load_checkpoint_encoder(args.checkpoint, converter_compatible=True)
    model.eval()
    example = validation[0]
    example_audio = example["audio"].unsqueeze(0)
    example_imu = example["imu"].unsqueeze(0)
    exported = torch.export.export(model, (example_audio, example_imu), strict=True)
    model_destination = args.output_model.resolve()
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, model_destination)
    reloaded = torch.export.load(model_destination).module()

    audio_rows = []
    imu_rows = []
    reference_rows = []
    with torch.inference_mode():
        for index in range(len(validation)):
            sample = validation[index]
            audio = sample["audio"].unsqueeze(0)
            imu = sample["imu"].unsqueeze(0)
            output = reloaded(audio, imu)
            if tuple(output.shape) != (1, 32) or not torch.isfinite(output).all():
                raise ValueError("Reloaded PT2 returned an invalid output")
            audio_rows.append(sample["audio"].numpy().copy())
            imu_rows.append(sample["imu"].numpy().copy())
            reference_rows.append(output.numpy()[0].copy())
    arrays = {
        "audio": np.ascontiguousarray(np.stack(audio_rows), dtype=np.float32),
        "imu": np.ascontiguousarray(np.stack(imu_rows), dtype=np.float32),
        "mean": np.ascontiguousarray(np.stack(reference_rows), dtype=np.float32),
    }
    expected = {"audio": (16, 180, 40), "imu": (16, 720, 6), "mean": (16, 32)}
    for name, shape in expected.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise AssertionError(f"Invalid converter validation array {name}")
    validation_destination = args.validation_output.resolve()
    validation_destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(validation_destination, **arrays)
    print(f"Wrote {model_destination}")
    print(f"Wrote {validation_destination}")


if __name__ == "__main__":
    main()
