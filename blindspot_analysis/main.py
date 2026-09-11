import sys
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import entropy, wasserstein_distance
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pdb
from transformers import BertTokenizer, BertModel

# Load data
with open(os.path.join("/Users/cril/tanmoy/research/data", "chaosNLI", "embeddings", "snli.pkl"), 'rb') as f:
    snli = pickle.load(f)

with open(os.path.join("/Users/cril/tanmoy/research/data", "chaosNLI", "embeddings", "mnli_m.pkl"), 'rb') as f:
    mnli = pickle.load(f)

premise = np.concatenate((snli["premise"], mnli["premise"]), axis=0)
hypothesis = np.concatenate((snli["hypothesis"], mnli["hypothesis"]), axis=0)
label_dist = np.concatenate((snli["label_dist"], mnli["label_dist"]), axis=0)

# Step 1: Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Create set representations
combined_sets = []

for premise_text, hypothesis_text in zip(premise, hypothesis):
    premise_tokens = tokenizer(premise_text)
    hypothesis_tokens = tokenizer(hypothesis_text)

    premise_embedding_set=model(**premise_tokens)
    hypothesis_embedding_set=model(**hypothesis_tokens)

    # Combine premise and hypothesis embeddings
    combined_set = premise_embedding_set + hypothesis_embedding_set
    combined_sets.append(combined_set)


# Determine the maximum set length (you can set a fixed max length to limit)
max_set_length = max(len(s) for s in combined_sets)
max_set_length = min(max_set_length, 50)  # For example, limit to 50 tokens

def pad_sequences(embedding_sets, max_length, embedding_dim):
    padded_sets = []
    set_lengths = []
    for embedding_set in embedding_sets:
        set_length = len(embedding_set)
        set_lengths.append(min(set_length, max_length))
        if set_length < max_length:
            padding = [np.zeros(embedding_dim) for _ in range(max_length - set_length)]
            padded_set = embedding_set + padding
        else:
            padded_set = embedding_set[:max_length]
        padded_sets.append(padded_set)
    return np.array(padded_sets), np.array(set_lengths)

X_padded, set_lengths = pad_sequences(combined_sets, max_set_length, embedding_dim)
X_padded = torch.tensor(X_padded, dtype=torch.float32).to(device)  # Shape: (num_examples, max_set_length, embedding_dim)
y = torch.tensor(label_dist, dtype=torch.float32).to(device)       # Shape: (num_examples, num_classes)
set_lengths = torch.tensor(set_lengths, dtype=torch.long).to(device)  # Shape: (num_examples,)

# Prepare data loaders
from torch.utils.data import Dataset, DataLoader

class NliDataset(Dataset):
    def __init__(self, X, set_lengths, y):
        self.X = X
        self.set_lengths = set_lengths
        self.y = y
        
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        return self.X[idx], self.set_lengths[idx], self.y[idx]
    
    # Split the data
X_train, X_test, lengths_train, lengths_test, y_train, y_test = train_test_split(
    X_padded.cpu(), set_lengths.cpu(), y.cpu(), test_size=500, random_state=2024+exp_seed)
X_train, X_calib, lengths_train, lengths_calib, y_train, y_calib = train_test_split(
    X_train, lengths_train, y_train, test_size=500, random_state=2024+exp_seed)

# Convert back to tensors and move to device
X_train = X_train.to(device)
lengths_train = lengths_train.to(device)
y_train = y_train.to(device)
X_calib = X_calib.to(device)
lengths_calib = lengths_calib.to(device)
y_calib = y_calib.to(device)
X_test = X_test.to(device)
lengths_test = lengths_test.to(device)
y_test = y_test.to(device)
# Create datasets and data loaders
train_dataset = NliDataset(X_train, y_train)
calib_dataset = NliDataset(X_calib, y_calib)
test_dataset = NliDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
calib_loader = DataLoader(calib_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

embedding_dim = X_padded.shape[2]
hidden_dim = 128
output_dim = 64  # Dimension of the distribution embedding


# Define the model
class DeepSets(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepSets, self).__init__()
        # Phi network processes individual elements
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Rho network processes aggregated set representation
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, set_lengths):
        # x: (batch_size, max_set_length, input_dim)
        # set_lengths: (batch_size,)
        batch_size = x.size(0)
        
        # Apply phi to each element
        x = self.phi(x)  # (batch_size, max_set_length, hidden_dim)
        
        # Mask padding positions
        mask = torch.arange(max_set_length).expand(batch_size, max_set_length).to(device)
        mask = mask < set_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # (batch_size, max_set_length, 1)
        x = x * mask.float()
        
        # Aggregate using sum
        x = torch.sum(x, dim=1)  # (batch_size, hidden_dim)
        
        # Apply rho
        x = self.rho(x)  # (batch_size, output_dim)
        return x


def compute_mmd_loss(x1, x2, kernel='rbf', sigma=1.0):
    # x1, x2: (batch_size, embedding_dim)
    if kernel == 'rbf':
        # Compute pairwise distances
        x1_square = x1.pow(2).sum(dim=1, keepdim=True)
        x2_square = x2.pow(2).sum(dim=1, keepdim=True)
        xy = x1 @ x2.t()
        distances = x1_square + x2_square.t() - 2 * xy
        
        k = torch.exp(-distances / (2 * sigma ** 2))
        mmd = k.mean()
    else:
        raise ValueError('Unsupported kernel type')
    return mmd


def contrastive_mmd_loss(z_i, z_j, z_k, sigma=1.0):
    # z_i, z_j: Positive pair embeddings (batch_size, embedding_dim)
    # z_k: Negative examples embeddings (batch_size, embedding_dim)
    mmd_pos = compute_mmd_loss(z_i, z_j, sigma=sigma)
    mmd_neg = compute_mmd_loss(z_i, z_k, sigma=sigma)
    loss = mmd_neg - mmd_pos
    return loss


model = DeepSets(input_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
sigma = 1.0  # Kernel bandwidth for MMD

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        X_batch, lengths_batch, y_batch = batch
        batch_size = X_batch.size(0)
        
        # Forward pass
        z_i = model(X_batch, lengths_batch)  # Embeddings of the sets
        
        # Create positive pairs (shift embeddings by one)
        z_j = torch.roll(z_i, shifts=-1, dims=0)
        
        # Create negative pairs (shuffle embeddings)
        indices = torch.randperm(batch_size)
        z_k = z_i[indices]
        
        # Compute contrastive MMD loss
        loss = contrastive_mmd_loss(z_i, z_j, z_k, sigma=sigma)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)

# Get embeddings for training data
model.eval()
with torch.no_grad():
    z_train = model(X_train, lengths_train)  # (num_examples, output_dim)
    z_calib = model(X_calib, lengths_calib)
    z_test = model(X_test, lengths_test)

# Create datasets for classifier
train_dataset = torch.utils.data.TensorDataset(z_train, y_train)
calib_dataset = torch.utils.data.TensorDataset(z_calib, y_calib)
test_dataset = torch.utils.data.TensorDataset(z_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
calib_loader = DataLoader(calib_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


#encoded_input = tokenizer(text, return_tensors='pt')