"""Model components for image uncertainty and supervised-LVM experiments."""

from .backbones import BackboneInfo, FrozenVisualBackbone, build_backbone

__all__ = ["BackboneInfo", "FrozenVisualBackbone", "build_backbone"]
