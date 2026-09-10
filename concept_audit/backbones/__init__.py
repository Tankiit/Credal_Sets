"""Reuse the existing frozen visual extraction boundary."""
from models.backbones import FrozenVisualBackbone, build_backbone

__all__ = ["FrozenVisualBackbone", "build_backbone"]
