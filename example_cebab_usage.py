"""
Example usage of CEBaB dataset loader.

This script demonstrates how to:
1. Login to Hugging Face
2. Load the CEBaB dataset
3. Use the DataLoaders with all specified fields
"""

from load_ambigqa_dataset import create_cebab_dataloaders, login_to_huggingface, inspect_dataset_sample

# Step 1: Login to Hugging Face (if not already logged in)
# Option A: Use CLI login (recommended)
# Run: huggingface-cli login

# Option B: Login programmatically
# login_to_huggingface(token="your_token_here")

# Option C: Set environment variable
# export HF_TOKEN=your_token_here

# Step 2: Create CEBaB DataLoaders
dataloaders = create_cebab_dataloaders(
    dataset_name="CEBaB/CEBaB",
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

# Step 3: Use the DataLoaders
# Access different splits
if 'train' in dataloaders:
    train_loader = dataloaders['train']
    
    # Example: Training loop
    for i, batch in enumerate(train_loader):
        if i >= 2:
            break
        
        # Extract fields
        ids = batch['id']  # Format: "original_id_edit_id"
        original_ids = batch['original_id']
        edit_ids = batch['edit_id']
        is_original = batch['is_original']
        descriptions = batch['description']
        edit_goals = batch['edit_goal']
        edit_types = batch['edit_type']
        
        # Review labels
        review_majorities = batch['review_majority']
        review_distributions = batch['review_label_distribution']
        
        # Aspect labels
        food_majorities = batch['food_aspect_majority']
        ambiance_majorities = batch['ambiance_aspect_majority']
        service_majorities = batch['service_aspect_majority']
        noise_majorities = batch['noise_aspect_majority']
        
        # Metadata
        opentable_metadata = batch['opentable_metadata']

if 'validation' in dataloaders or 'val' in dataloaders:
    val_split = 'validation' if 'validation' in dataloaders else 'val'
    val_loader = dataloaders[val_split]

if 'test' in dataloaders:
    test_loader = dataloaders['test']

