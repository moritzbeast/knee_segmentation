import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from args import get_args
from dataset import KneeDataset, read_xray
from model import UNetLext
from utils import compute_mean_std, postprocess, dice_np

args = get_args()
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# 1: load data + model
train_set = pd.read_csv(os.path.join(args.csv_dir, "train.csv"))
val_set = pd.read_csv(os.path.join(args.csv_dir, "val.csv"))
test_set = pd.read_csv(os.path.join(args.csv_dir, "test.csv"))
mean, std = compute_mean_std(train_set)   # same normalization as in training

model = UNetLext(depth=args.depth, input_channels=1, output_channels=1)
model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pth"), map_location=device))
model.to(device).eval()

# 2: validation -> dice per image, raw and post-processed
val_loader = DataLoader(KneeDataset(val_set, mean, std), args.bs, shuffle=False)
dices_raw, dices_post = [], []

with torch.no_grad():
    for batch in val_loader:
        logits = model(batch["xray"].to(device))
        preds = (torch.sigmoid(logits[:, 0]) > 0.5).cpu().numpy()
        masks = batch["mask"][:, 0].numpy().astype(bool)

        for pred, mask in zip(preds, masks):
            dices_raw.append(dice_np(pred, mask))
            dices_post.append(dice_np(postprocess(pred).astype(bool), mask))

val_results = pd.DataFrame({
    "xray": [os.path.basename(p) for p in val_set["xrays"]],
    "dice_raw": dices_raw,
    "dice_post": dices_post,
})
val_results.to_csv(os.path.join(args.out_dir, "val_dice.csv"), index=False)

print(f"Val Dice raw:  {np.mean(dices_raw):.4f} ± {np.std(dices_raw):.4f}")
print(f"Val Dice post: {np.mean(dices_post):.4f} ± {np.std(dices_post):.4f}")

# 3: test -> predict every image (no masks, so read xrays directly instead of KneeDataset)
names, xrays, preds_raw, preds_post = [], [], [], []
with torch.no_grad():
    for path in test_set["xrays"]:
        xray = read_xray(path)                                              # (1, H, W), 0-1
        x = torch.from_numpy((xray - mean) / std).float()[None].to(device)  # (1, 1, H, W)
        pred = (torch.sigmoid(model(x))[0, 0] > 0.5).cpu().numpy()

        names.append(os.path.basename(path))
        xrays.append(xray[0])
        preds_raw.append(pred)
        preds_post.append(postprocess(pred).astype(bool))

# 4: test overview -> post-processed prediction of every test image, 30 per figure
per_fig, n_cols = 30, 6
for k, start in enumerate(range(0, len(xrays), per_fig)):
    idxs = range(start, min(start + per_fig, len(xrays)))
    n_rows = int(np.ceil(len(idxs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, i in zip(axes.flat, idxs):
        ax.imshow(xrays[i], cmap="gray")
        ax.contour(preds_post[i], levels=[0.5], colors="lime", linewidths=0.8)
        ax.set_title(names[i], fontsize=8)

    fig.suptitle("Test predictions (post-processed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"predictions_test_{k + 1}.png"), dpi=150)
    plt.close(fig)

# 5: which test images were changed by post-processing?
changed = [i for i in range(len(xrays)) if (preds_raw[i] != preds_post[i]).any()]
print(f"Post-processing changed {len(changed)} of {len(xrays)} test images")
for i in changed:
    removed = (preds_raw[i] & ~preds_post[i]).sum()
    filled = (preds_post[i] & ~preds_raw[i]).sum()
    print(f"  {names[i]}: {removed} px removed, {filled} px filled")

# plot only the changed images, 5 per figure, left = raw, right = post-processed
per_fig = 5
for k, start in enumerate(range(0, len(changed), per_fig)):
    idxs = changed[start:start + per_fig]
    fig, axes = plt.subplots(len(idxs), 2, figsize=(6, 3 * len(idxs)), squeeze=False)

    for row, i in zip(axes, idxs):
        for ax, pred, name, show_changes in [(row[0], preds_raw[i], "raw", False),
                                             (row[1], preds_post[i], "post-processed", True)]:
            ax.imshow(xrays[i], cmap="gray")
            overlay = np.zeros((*pred.shape, 4))                           # RGBA, transparent by default
            overlay[pred] = [0, 1, 0, 0.3]                                 # prediction
            if show_changes:
                overlay[preds_raw[i] & ~preds_post[i]] = [1, 0, 0, 0.8]    # removed island
                overlay[preds_post[i] & ~preds_raw[i]] = [0, 0, 1, 0.8]    # filled hole
            ax.imshow(overlay)
            ax.set_title(f"{names[i]} {name}", fontsize=9)
            ax.axis("off")

    fig.suptitle("green = prediction | right only: red = removed island, blue = filled hole", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"postprocessing_test_{k + 1}.png"), dpi=150)
    plt.close(fig)
