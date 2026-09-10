"""Replaceable frozen visual backbones for image feature extraction.

This module is intentionally independent from the downstream uncertainty / CBM code.
It exposes one small contract:

    backbone = build_backbone(kind="hf", model_name="facebook/dinov2-base", device="cuda")
    features = backbone.encode_pil(images)  # (B, D)

The rest of the codebase only consumes the resulting feature matrix, so DINOv2,
DINOv3, ResNet, ConvNeXt, ViT, etc. can be swapped without changing analysis code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class BackboneInfo:
    kind: str
    model_name: str
    feature_dim: int | None
    frozen: bool = True


class FrozenVisualBackbone(ABC):
    """Common interface for frozen global-image feature extractors."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = torch.device(device)

    @abstractmethod
    def encode_pil(self, images: Sequence) -> torch.Tensor:
        """Return float32 global embeddings with shape ``(batch, feature_dim)``."""

    @property
    @abstractmethod
    def feature_dim(self) -> int | None:
        pass

    @property
    @abstractmethod
    def info(self) -> BackboneInfo:
        pass


def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class HuggingFaceBackbone(FrozenVisualBackbone):
    """Frozen Hugging Face vision encoder using its native image processor.

    Pooling policy:
      1. ``pooler_output`` when provided;
      2. otherwise CLS token ``last_hidden_state[:, 0]``;
      3. otherwise mean-pool spatial features.

    This covers DINOv2/DINOv3 and many ViT-like models.
    """

    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device)
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = _freeze(AutoModel.from_pretrained(model_name).to(self.device))
        self._feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self) -> int | None:
        cfg = getattr(self.model, "config", None)
        for key in ("hidden_size", "embed_dim", "projection_dim"):
            value = getattr(cfg, key, None) if cfg is not None else None
            if value is not None:
                return int(value)
        return None

    def encode_pil(self, images: Sequence) -> torch.Tensor:
        batch = self.processor(images=list(images), return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = self.model(**batch)

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled.float()

        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError(
                f"{self.model_name} returned neither pooler_output nor last_hidden_state"
            )
        if hidden.ndim == 3:
            return hidden[:, 0].float()
        if hidden.ndim == 4:
            return hidden.flatten(2).mean(-1).float()
        raise RuntimeError(f"Unsupported hidden-state shape: {tuple(hidden.shape)}")

    @property
    def feature_dim(self) -> int | None:
        return self._feature_dim

    @property
    def info(self) -> BackboneInfo:
        return BackboneInfo("hf", self.model_name, self.feature_dim)


class TimmBackbone(FrozenVisualBackbone):
    """Frozen ``timm`` image backbone.

    ``num_classes=0`` exposes the global feature vector rather than classifier
    logits. Native pretrained preprocessing is resolved from timm itself.
    """

    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device)
        try:
            import timm
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as exc:
            raise ImportError("Install timm to use --backbone timm") from exc

        self.model = _freeze(
            timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device)
        )
        data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(**data_config, is_training=False)
        self._feature_dim = int(getattr(self.model, "num_features", 0)) or None

    def encode_pil(self, images: Sequence) -> torch.Tensor:
        pixel_values = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.inference_mode():
            features = self.model(pixel_values)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.flatten(2).mean(-1)
        return features.float()

    @property
    def feature_dim(self) -> int | None:
        return self._feature_dim

    @property
    def info(self) -> BackboneInfo:
        return BackboneInfo("timm", self.model_name, self.feature_dim)


def build_backbone(kind: str, model_name: str, device: str) -> FrozenVisualBackbone:
    kind = kind.lower()
    if kind == "hf":
        return HuggingFaceBackbone(model_name=model_name, device=device)
    if kind == "timm":
        return TimmBackbone(model_name=model_name, device=device)
    raise ValueError(f"Unknown backbone kind: {kind!r}. Expected one of: hf, timm")
