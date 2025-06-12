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










# If using a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

