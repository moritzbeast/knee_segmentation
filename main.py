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
from torchvision.transforms import v2

# 1: We need arguments
args = get_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# 2: Loading the csv file
train_set = pd.read_csv(os.path.join(args.csv_dir, "train.csv"))
val_set = pd.read_csv(os.path.join(args.csv_dir, "val.csv"))

train_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.RandomInvert(p=0.2),
])

# 3: create dataset
mean, std = compute_mean_std(train_set)
train_dataset = KneeDataset(train_set, mean, std, train_transform)
val_dataset = KneeDataset(val_set, mean, std)

# 4: create loaders
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