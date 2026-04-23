"""
Dataset loader registry.

    from loaders import LOADERS
    bundle = LOADERS['cebab'].load(
        tokenizer_name='distilbert-base-uncased',
        batch_size=8,
        max_length=256,
    )
"""
from loaders.interface import DatasetBundle, DatasetLoader
from loaders import cebab, hatexplain, goemotions, sst2

LOADERS: dict[str, DatasetLoader] = {
    "cebab":       cebab.LOADER,
    "hatexplain":  hatexplain.LOADER,
    "goemotions":  goemotions.LOADER,
    "sst2":        sst2.LOADER,
}

__all__ = ["DatasetBundle", "DatasetLoader", "LOADERS"]