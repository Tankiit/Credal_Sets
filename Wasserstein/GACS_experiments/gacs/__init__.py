# GACS - Geometry-Aware Credal Predictive Sets

# Silence Hugging Face tokenizers fork/parallelism warnings by default.
# This must be set before importing `transformers` or creating tokenizers.
import os as _os
if "TOKENIZERS_PARALLELISM" not in _os.environ:
    _os.environ["TOKENIZERS_PARALLELISM"] = "false"
