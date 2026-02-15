"""
Simple example of using the AmbigQA Star dataset loader.

This script demonstrates how to:
1. Login to Hugging Face
2. Load the dataset
3. Use the DataLoaders with different IDs
"""

from load_ambigqa_dataset import create_dataloaders, login_to_huggingface, inspect_dataset_sample

# Step 1: Login to Hugging Face
# Option A: Use CLI login (recommended)
# Run: huggingface-cli login
# Then the script will detect you're already logged in

# Option B: Login programmatically with token
# login_to_huggingface(token="your_token_here")

# Option C: Set environment variable
# export HF_TOKEN=your_token_here

# Step 2: Create DataLoaders
dataloaders = create_dataloaders(
    dataset_name="ttomov/ambigqa_star",
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

# Step 3: Use the DataLoaders
# Access different splits
if 'train' in dataloaders:
    train_loader = dataloaders['train']
    
    # Iterate through batches
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        
        # Access specific fields (adjust based on actual dataset structure)
        for key in batch.keys():
            if key != 'id':
                value = batch[key]

if 'validation' in dataloaders or 'val' in dataloaders:
    val_split = 'validation' if 'validation' in dataloaders else 'val'
    val_loader = dataloaders[val_split]

if 'test' in dataloaders:
    test_loader = dataloaders['test']

