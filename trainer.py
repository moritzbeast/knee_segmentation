import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from args import get_args
from utils import dice_loss_from_logits, dice_score_from_logits

args = get_args()
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(device)


def train_model(model, train_loader, val_loader):
    model.to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # warmup, then cosine decay; steps once per batch
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                              steps_per_epoch=len(train_loader))

    os.makedirs(args.out_dir, exist_ok=True)
    best_dice = -1
    history = {key: [] for key in ["train_loss", "train_dice", "val_loss", "val_dice"]}

    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        model.train()
        running_loss = 0
        running_dice = 0

        for batch in train_loader:
            images = batch["xray"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = bce(outputs, masks) + dice_loss_from_logits(outputs, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            # weighted by batch size, the last batch is smaller (45 = 16 + 16 + 13)
            running_dice += dice_score_from_logits(outputs, masks) * images.size(0)

        train_loss = running_loss / len(train_loader)
        train_dice = running_dice / len(train_loader.dataset)

        val_loss, val_dice = validate_model(model, val_loader, bce)

        for key, value in [("train_loss", train_loss), ("train_dice", train_dice),
                           ("val_loss", val_loss), ("val_dice", val_dice)]:
            history[key].append(value)

        if val_dice > best_dice:
            best_dice = val_dice
            best_metrics = {"best_epoch": epoch + 1, "val_dice": val_dice, "val_loss": val_loss}
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}",
                         val_dice=f"{val_dice:.4f}", best=f"{best_dice:.4f}")

    tqdm.write(f"Best model: epoch {best_metrics['best_epoch']} | Val Dice: {best_dice:.4f}")
    plot_metrics(history)

    if args.plot_preds:
        # model in memory has the weights of the last epoch -> load the best one
        model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pth"), map_location=device))
        plot_predictions(model, val_loader)

    return best_metrics


def validate_model(model, val_loader, bce):
    model.eval()
    running_loss = 0
    running_dice = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["xray"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)

            loss = bce(outputs, masks) + dice_loss_from_logits(outputs, masks)
            running_loss += loss.item()
            running_dice += dice_score_from_logits(outputs, masks) * images.size(0)

    val_loss = running_loss / len(val_loader)
    val_dice = running_dice / len(val_loader.dataset)

    return val_loss, val_dice


def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmax(history["val_dice"])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (key, title) in zip(axes, [("loss", "Loss (BCE + Dice)"), ("dice", "Dice Score")]):
        ax.plot(epochs, history[f"train_{key}"], color="blue", linewidth=2, label="Training")
        ax.plot(epochs, history[f"val_{key}"], color="orange", linewidth=2, linestyle="--", label="Validation")
        ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1, label=f"Best model (epoch {best_epoch})")
        ax.set_title(f"{title} over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "metrics.png"), dpi=150)
    plt.close(fig)

def plot_predictions(model, val_loader, n_cols=5):
    model.eval()
    xrays, masks, preds, dices = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["xray"].to(device)
            targets = batch["mask"].to(device)
            logits = model(images)

            # dice per image, not per batch
            for i in range(images.size(0)):
                dices.append(dice_score_from_logits(logits[i:i+1], targets[i:i+1]))

            xrays.append(images[:, 0].cpu().numpy())
            masks.append(targets[:, 0].cpu().numpy().astype(bool))
            preds.append((torch.sigmoid(logits[:, 0]) > 0.5).cpu().numpy())

    xrays, masks, preds = np.concatenate(xrays), np.concatenate(masks), np.concatenate(preds)

    n_rows = int(np.ceil(len(xrays) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for ax in axes.flat:
        ax.axis("off")

    for ax, xray, mask, pred, dice in zip(axes.flat, xrays, masks, preds, dices):
        ax.imshow(xray, cmap="gray")
        error = np.zeros((*mask.shape, 4))     # RGBA, transparent by default
        error[mask & pred] = [0, 1, 0, 0.3]    # correct
        error[pred & ~mask] = [1, 0, 0, 0.7]   # false positive
        error[mask & ~pred] = [0, 0, 1, 0.7]   # false negative
        ax.imshow(error)
        ax.set_title(f"Dice {dice:.4f}", fontsize=9)

    fig.suptitle("Validation: green = correct, red = false positive, blue = false negative")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "predictions_val.png"), dpi=150)
    plt.close(fig)