# Training Cells Added to example.ipynb

## Summary
Successfully added 5 new cells to the end of `example.ipynb` for complete model training pipeline.

## New Cells Added:

### Cell 4: Markdown Header
- Title: "## Training Pipeline"
- Describes the training section

### Cell 5: Training Loop
**Features:**
- 10 epochs by default (configurable)
- Adam optimizer (lr=1e-3)
- Progress bars with tqdm
- Tracks 4 loss components:
  - Total loss
  - Concept loss
  - Task loss  
  - DRO loss
- Prints epoch summaries

### Cell 6: Visualization
**Plots:**
- Total loss (train & val)
- Concept loss (train & val)
- Task loss (train & val)
- DRO loss (train & val)
- Shows final statistics including best validation epoch

### Cell 7: Evaluation
**Metrics:**
- Task accuracy, F1 (macro & weighted)
- Per-concept accuracy for all 4 CEBaB concepts
- Confusion matrix visualization
- Credible width (epsilon) distribution analysis
- Statistics on credal set sizes by class

### Cell 8: Save Model
**Saves:**
- Model checkpoint (`credal_dro_cebab_model.pt`)
- Training history JSON (`credal_dro_cebab_history.json`)
- Config JSON (`credal_dro_cebab_config.json`)
- All saved in `./checkpoints/` directory

## How to Use:

1. Run the first 3 cells to load data, encoder, and initialize model
2. Run Cell 5 (Training Cell) to start training
3. Run Cell 6 to visualize training curves
4. Run Cell 7 to evaluate on test set
5. Run Cell 8 to save the trained model

## Notes:
- All cells use metadata-driven config (no hardcoded values)
- Compatible with MPS/CUDA/CPU devices
- Comprehensive error handling and progress tracking
