"""
SST-2 dataset loader.

SST-2 does NOT have human annotator labels. CREDENCE generates synthetic
annotator entropy using an LLM (see legacy/credence_acl/... for the
generation script). For the NeurIPS paper, two paths:

  (a) Use CREDENCE's pre-generated LLM annotator entropies if cached
  (b) Treat SST-2 as the "no AU signal" edge case: has_annotator_entropy=False,
      and the grad-iso probe will skip AU-related measurements on this dataset

This file implements (b) as the default; flip the flag once the LLM-generated
annotations are ready to ship.
"""
from __future__ import annotations

from loaders.interface import DatasetBundle, DatasetLoader

# TODO(PR5): implement SST-2 loading. For PR1, this raises.
# Options:
#   1. Use `datasets.load_dataset("stanfordnlp/sst2")` + HF tokenizer
#   2. If CREDENCE's LLM-annotation cache exists, load it and set
#      has_annotator_entropy=True


class _SST2Loader:
    def load(
        self,
        tokenizer_name: str,
        batch_size: int,
        max_length: int = 128,
        seed: int = 42,
        subset_fraction: float = 1.0,
        num_workers: int = 0,
    ) -> DatasetBundle:
        raise NotImplementedError(
            "SST-2 loader not yet implemented. Scheduled for PR 5 — needs "
            "decision on LLM-generated annotator entropies from CREDENCE "
            "versus treating SST-2 as the has_annotator_entropy=False edge "
            "case."
        )


LOADER: DatasetLoader = _SST2Loader()