"""Optional experiment logging helpers for TensorBoard and W&B."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


def _to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_dict(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_plain_dict(value.to_dict())
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested metric dictionaries into scalar key/value pairs."""
    flattened: Dict[str, float] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}{key}" if prefix else str(key)
        plain_value = _to_plain_dict(value)
        if isinstance(plain_value, Mapping):
            flattened.update(flatten_metrics(plain_value, prefix=f"{full_key}/"))
        elif isinstance(plain_value, (list, tuple)):
            continue
        else:
            try:
                flattened[full_key] = float(plain_value)
            except (TypeError, ValueError):
                continue
    return flattened


class ExperimentLogger:
    """Thin wrapper around optional TensorBoard and W&B loggers."""

    def __init__(
        self,
        log_dir: str | Path,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_mode: str = "offline",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._wandb_run = None

        if use_tensorboard and SummaryWriter is not None:
            self._writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        if use_wandb and wandb is not None:
            init_kwargs = {
                "project": wandb_project or "neurips-credal",
                "name": wandb_run_name,
                "dir": str(self.log_dir),
                "mode": wandb_mode,
            }
            self._wandb_run = wandb.init(**{key: value for key, value in init_kwargs.items() if value is not None})

    @property
    def enabled(self) -> bool:
        return self._writer is not None or self._wandb_run is not None

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        plain_params = _to_plain_dict(params)
        if self._writer is not None:
            self._writer.add_text("hparams", json.dumps(plain_params, indent=2, default=str))
        if self._wandb_run is not None:
            self._wandb_run.config.update(plain_params, allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, Any], step: int, prefix: str = "") -> None:
        flattened = flatten_metrics(metrics, prefix=prefix)
        if self._writer is not None:
            for key, value in flattened.items():
                self._writer.add_scalar(key, value, global_step=step)
        if self._wandb_run is not None:
            self._wandb_run.log(flattened, step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
