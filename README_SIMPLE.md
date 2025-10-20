# CLARITY: Credal Concept Bottleneck Models

Implementation of Credal Concept Bottleneck Models with epistemic and aleatoric uncertainty quantification for text classification.

## Overview

This repository contains the implementation of Credal-CBM, which separates epistemic and aleatoric uncertainty using credal sets and ensemble methods for interpretable text classification.

## Key Components

**Core Model:**
- `credal_cbm_model.py` - Main model with uncertainty quantification
- `embedding.py` - RKHS embedding utilities
- `second_order.py` - Second-order uncertainty computation

**Training:**
- `run_sst2_experiments.py` - SST-2 sentiment analysis
- `run_cebab_experiments.py` - CEBAB multi-aspect sentiment

**Demonstrations:**
- `application_demonstrator.py` - Active learning and intervention demos
- `standalone_active_learning_demo.py` - Synthetic active learning demo

## Installation

```bash
pip install torch transformers datasets numpy scipy matplotlib seaborn pandas scikit-learn tqdm
```

## Usage

Train on SST-2:
```bash
python run_sst2_experiments.py --output_dir ./results_sst2
```

Train on CEBAB:
```bash
python run_cebab_experiments.py --output_dir ./results_cebab
```

Generate active learning demo:
```bash
python application_demonstrator.py --checkpoint results_sst2/credal_cbm_sst2.pt --dataset sst2 --output_dir ./demos
```

## Results

**Active Learning Efficiency:**
- Epistemic-guided sampling achieves 1.2x sample efficiency vs random sampling
- Saves 20% labels to reach target accuracy on SST-2

**Model Performance:**
- SST-2: 84.4% test accuracy
- CEBAB: 84.7% test accuracy

## Citation

If you use this code, please cite our work.

## License

MIT License
