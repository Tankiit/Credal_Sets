# Training for 50 Epochs - Complete Implementation

## Summary

Updated `test_ternary.py` to run for 50 epochs with comprehensive plotting of all loss components including the new aleatoric head.

## Changes Made

### 1. Updated Epoch Count
```python
num_epochs = 50  # Changed from 10 to 50
```

### 2. Added Comprehensive Plotting

**Six plots showing:**
1. **Total Loss** - Train vs validation loss over epochs
2. **Task Loss** - Cross-entropy on sentiment classification
3. **Concept Loss** - Ternary concept supervision (CrossEntropy)
4. **Aleatoric Loss** - MSE on aleatoric uncertainty prediction
5. **Robust Loss** - DRO worst-case loss
6. **Validation Accuracy** - Best accuracy marked with red dashed line

### 3. Training History Tracking

```python
history = {
    'train_loss': [],
    'train_concept_loss': [],
    'train_task_loss': [],
    'train_robust_loss': [],
    'train_aleatoric_loss': [],  # NEW
    'val_loss': [],
    'val_acc': [],
}
```

### 4. Enhanced Progress Display

**Training loop shows:**
- Total loss
- Task loss
- Concept loss
- Robust loss
- Aleatoric loss (NEW)
- Validation accuracy

**Example output:**
```
Epoch 10/50 Summary:
  Train - Total: 2.1234 | Concept: 0.8234 | Task: 0.9123 | Robust: 0.6234 | Ale: 0.1456
  Val   - Loss: 2.2345 | Acc: 0.4234
```

## Running the Training

```bash
python test_ternary.py 2>&1 | tee training_50epochs.log
```

This will:
- Train for 50 epochs
- Display progress every 10 epochs
- Save all output to `training_50epochs.log`
- Generate `training_results_50epochs.png` with 6 plots
- Display plots at the end

## Output Files

1. **training_50epochs.log** - Complete training log
2. **training_results_50epochs.png** - 6-plot figure with:
   - Total loss (train/val)
   - Task loss
   - Concept loss
   - Aleatoric loss
   - Robust loss
   - Validation accuracy with best epoch marked

## Expected Training Behavior

### Loss Components

1. **Task Loss**: Should decrease as model learns sentiment classification
2. **Concept Loss**: Should decrease as model learns ternary concepts
3. **Aleatoric Loss**: Should decrease as model learns to predict annotator uncertainty
4. **Robust Loss**: May fluctuate due to adversarial PGD inner loop
5. **Total Loss**: Weighted sum of all components

### Validation Accuracy

- Should generally increase over epochs
- May have some fluctuations due to small dataset size
- Best epoch marked with red dashed line

## Model Configuration

```python
CredalDROConfig(
    num_concepts=4,
    num_classes=3,
    concept_classes=3,  # Ternary
    n_heads=5,
    lambda_concept=1.0,
    lambda_dro=0.1,
    beta_width=0.01,
    use_aleatoric=True,  # Aleatoric head enabled
    lambda_ale=1.0,      # Aleatoric loss weight
)
```

## Complete Loss Function

```
L_total = L_task + λ_c·L_concept + λ_dro·L_robust + β·Ω(Σ) + λ_ale·L_ale
```

Where:
- `L_task`: Cross-entropy on sentiment labels
- `L_concept`: Cross-entropy on ternary concepts
- `L_robust`: DRO worst-case loss
- `Ω(Σ)`: Width penalty on credal set
- `L_ale`: MSE on aleatoric uncertainty (NEW!)

## Monitoring Training

### Every 10 Epochs
The script prints:
- Train losses for all components
- Validation loss and accuracy
- Running best accuracy

### At the End
- Final statistics summary
- Test evaluation
- Plots saved and displayed

## Interpreting Results

### Good Training Signs
- ✅ Total loss decreasing
- ✅ Validation accuracy increasing
- ✅ All loss components stabilizing
- ✅ Aleatoric loss decreasing (model learns uncertainty)

### Potential Issues
- ⚠️ Overfitting: Val loss increases while train loss decreases
- ⚠️ Instability: Large fluctuations in robust loss
- ⚠️ Aleatoric not learning: Loss stays high

## File Locations

After training completes:
```
Wasserstein/
├── test_ternary.py                # Updated script (50 epochs)
├── training_50epochs.log          # Complete log
├── training_results_50epochs.png  # Plots
└── training_summary.txt           # Final stats (if saved)
```

## Status

✅ **Training started in background**
- Running for 50 epochs
- Will generate comprehensive plots
- All loss components tracked

The training will take some time to complete. You can monitor progress in `training_50epochs.log`!
