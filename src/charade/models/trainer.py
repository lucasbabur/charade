"""Mini-batch trainer with in-training validation and early stopping on validation log loss."""

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from charade.features.spec import Encoded


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4096
    max_epochs: int = 6
    evals_per_epoch: int = 4
    patience: int = 4
    id_dropout: float = 0.0
    id_dropout_field: int | None = None
    seed: int = 0
    fixed_steps: int | None = None
    """Train exactly this many steps with no validation (refit on train + val with a step count chosen earlier)."""


@dataclass
class TrainResult:
    """Best model (by validation log loss) and the learning curve."""

    model: nn.Module
    best_val_logloss: float
    best_step: int
    history: list[tuple[int, float, float]] = field(default_factory=list[tuple[int, float, float]])
    seconds: float = 0.0


def device() -> torch.device:
    """CUDA when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensors(data: Encoded, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(data.categorical).to(dev), torch.from_numpy(data.dense).to(dev)


@torch.no_grad()
def predict_logits(model: nn.Module, data: Encoded, batch_size: int = 65536) -> np.ndarray:
    """Logits for every row (eval mode)."""
    dev = next(model.parameters()).device
    model.eval()
    cat, dense = _tensors(data, dev)
    out = [model(cat[i : i + batch_size], dense[i : i + batch_size]) for i in range(0, len(cat), batch_size)]
    return torch.cat(out).float().cpu().numpy().astype(np.float64)


def _val_logloss(model: nn.Module, cat: torch.Tensor, dense: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(cat[i : i + 65536], dense[i : i + 65536]) for i in range(0, len(cat), 65536)])
        return float(nn.functional.binary_cross_entropy_with_logits(logits, y).item())


def train(
    model: nn.Module,
    train_data: Encoded,
    y_train: np.ndarray,
    val_data: Encoded,
    y_val: np.ndarray,
    config: TrainConfig,
) -> TrainResult:
    """Train with AdamW; keep the parameters with the best validation log loss."""
    start = time.monotonic()
    torch.manual_seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    dev = device()
    model = model.to(dev)
    cat, dense = _tensors(train_data, dev)
    y = torch.from_numpy(y_train.astype(np.float32)).to(dev)
    v_cat, v_dense = _tensors(val_data, dev)
    v_y = torch.from_numpy(y_val.astype(np.float32)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    n = len(y)
    steps_per_epoch = (n + config.batch_size - 1) // config.batch_size
    eval_every = max(1, steps_per_epoch // config.evals_per_epoch)
    best, best_step, best_state, stale, step = float("inf"), 0, copy.deepcopy(model.state_dict()), 0, 0
    history: list[tuple[int, float, float]] = []
    running = 0.0
    for _ in range(config.max_epochs):
        order = torch.randperm(n, generator=generator).to(dev)
        for batch_start in range(0, n, config.batch_size):
            index = order[batch_start : batch_start + config.batch_size]
            batch_cat = cat[index]
            if config.id_dropout and config.id_dropout_field is not None:
                drop = torch.rand(len(index), device=dev) < config.id_dropout
                batch_cat[:, config.id_dropout_field] = torch.where(drop, 0, batch_cat[:, config.id_dropout_field])
            model.train()
            loss = nn.functional.binary_cross_entropy_with_logits(model(batch_cat, dense[index]), y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            step += 1
            if config.fixed_steps is not None:
                if step >= config.fixed_steps:
                    return TrainResult(model, float("nan"), step, history, time.monotonic() - start)
                continue
            if step % eval_every == 0:
                val = _val_logloss(model, v_cat, v_dense, v_y)
                history.append((step, running / eval_every, val))
                running = 0.0
                if val < best - 1e-6:
                    best, best_step, best_state, stale = val, step, copy.deepcopy(model.state_dict()), 0
                else:
                    stale += 1
                if stale >= config.patience:
                    model.load_state_dict(best_state)
                    return TrainResult(model, best, best_step, history, time.monotonic() - start)
    if config.fixed_steps is not None:
        return TrainResult(model, float("nan"), step, history, time.monotonic() - start)
    model.load_state_dict(best_state)
    return TrainResult(model, best, best_step, history, time.monotonic() - start)
