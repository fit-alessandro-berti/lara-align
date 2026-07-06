from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lara_align.model import LARANeuralModel


def build_model(model_config: dict[str, Any]) -> LARANeuralModel:
    config = dict(model_config)
    # Checkpoints written before bidirectional arc messages were trained with
    # forward edge types only; rebuilding them with 2 relation maps makes the
    # model ignore the reverse edge types (2 and 3) exactly as it did then.
    config.setdefault("num_edge_types", 2)
    return LARANeuralModel(**config)


def save_checkpoint(
    path: str | Path,
    model: LARANeuralModel,
    model_config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[LARANeuralModel, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint
