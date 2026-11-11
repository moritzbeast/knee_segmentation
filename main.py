from args import get_args
import pandas as pd
import os
from dataset import KneeDataset
from torch.utils.data import DataLoader
from model import UNetLext
import random
import numpy as np
import torch
from utils import compute_mean_std
from trainer import train_model

# 1: We need arguments
args = get_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# 2: Loading the csv file
train_set = pd.read_csv(os.path.join(args.csv_dir, "train.csv"))
val_set = pd.read_csv(os.path.join(args.csv_dir, "val.csv"))

# 3: 
mean, std = compute_mean_std(train_set)
train_dataset = KneeDataset(train_set, mean, std)
val_dataset = KneeDataset(val_set, mean, std)

# 4:
train_loader = DataLoader(train_dataset, args.bs, shuffle=True)
val_loader = DataLoader(val_dataset, args.bs, shuffle=False)

# 5: Initialize the model
model = UNetLext(
    depth=args.depth,
    input_channels=1,
    output_channels=1
)

# 6: Train model
train_model(model, train_loader, val_loader)