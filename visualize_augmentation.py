import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import tv_tensors
from torchvision.transforms import v2
from dataset import read_xray, read_mask

torch.manual_seed(0)

xray = tv_tensors.Image(torch.from_numpy(read_xray("data/xrays/0.png")))
mask = tv_tensors.Mask(torch.from_numpy(read_mask("data/masks/0.png")))

# each transformation on its own (same values as in main.py), plus all combined
transforms = {
    "Horizontal flip": v2.RandomHorizontalFlip(p=1.0),
    "Rotation ±10°": v2.RandomAffine(degrees=10),
    "Translation 5%": v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    "Scale 0.9-1.1": v2.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    "Brightness/contrast": v2.ColorJitter(brightness=0.2, contrast=0.2),
    "All combined": v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ]),
}

n = 4
fig, axes = plt.subplots(len(transforms), n + 1, figsize=(2.5 * (n + 1), 2.5 * len(transforms)))

for row, (name, t) in zip(axes, transforms.items()):
    samples = [(xray, mask)] + [t(xray, mask) for _ in range(n)]

    for ax, (x, m) in zip(row, samples):
        # fixed 0-1 range, otherwise imshow rescales and hides brightness changes
        ax.imshow(x[0], cmap="gray", vmin=0, vmax=1)
        ax.contour(m[0], levels=[0.5], colors="lime", linewidths=0.8)
        ax.axis("off")

    row[0].set_title("Original", fontsize=8)
    row[1].set_title(name, fontsize=8)

fig.tight_layout()
fig.savefig("data/outputs/augmentations.png", dpi=150)
plt.close(fig)
