import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import AgreementModel
from .objective import objective


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_batch(model: AgreementModel, optimizer: torch.optim.Optimizer, batch: dict[str, Any]) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, parts = objective(model, batch["image"], batch["text"], batch["clinical"], batch["label"], list(batch["silo"]))
    total.backward()
    optimizer.step()
    return {"total": float(total.detach()), **{name: float(value.detach()) for name, value in parts.items()}}


def save_state(path: str | Path, model: AgreementModel, optimizer: torch.optim.Optimizer, seed: int, step: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "step": step, "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    fd, name = tempfile.mkstemp(prefix=".state-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_state(path: str | Path, model: AgreementModel, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return {"seed": state["seed"], "step": state["step"]}
