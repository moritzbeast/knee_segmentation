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
from utils import dice_loss_from_logits, dice_score_from_logits, postprocess, dice_np


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
            # weighted by batch size, the last batch can be smaller
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

def plot_predictions(model, val_loader, per_fig=5):
    model.eval()
    xrays, masks, preds_raw = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["xray"].to(device)
            logits = model(images)

            xrays.append(images[:, 0].cpu().numpy())
            masks.append(batch["mask"][:, 0].numpy().astype(bool))
            preds_raw.append((torch.sigmoid(logits[:, 0]) > 0.5).cpu().numpy())

    xrays, masks, preds_raw = np.concatenate(xrays), np.concatenate(masks), np.concatenate(preds_raw)

    # remove islands + fill holes on every prediction
    preds_post = np.stack([postprocess(p) for p in preds_raw]).astype(bool)

    dices_raw = [dice_np(p, m) for p, m in zip(preds_raw, masks)]
    dices_post = [dice_np(p, m) for p, m in zip(preds_post, masks)]
    print(f"Mean val Dice: raw {np.mean(dices_raw):.4f} | post-processed {np.mean(dices_post):.4f}")

    # one figure per 5 images, left = raw, right = post-processed
    for k, start in enumerate(range(0, len(xrays), per_fig)):
        idxs = range(start, min(start + per_fig, len(xrays)))
        fig, axes = plt.subplots(len(idxs), 2, figsize=(6, 3 * len(idxs)), squeeze=False)

        for row, i in zip(axes, idxs):
            for ax, pred, dice, name in [(row[0], preds_raw[i], dices_raw[i], "raw"),
                                         (row[1], preds_post[i], dices_post[i], "post-processed")]:
                ax.imshow(xrays[i], cmap="gray")
                error = np.zeros((*masks[i].shape, 4))      # RGBA, transparent by default
                error[masks[i] & pred] = [0, 1, 0, 0.3]     # correct
                error[pred & ~masks[i]] = [1, 0, 0, 0.7]    # false positive
                error[masks[i] & ~pred] = [0, 0, 1, 0.7]    # false negative
                ax.imshow(error)
                ax.set_title(f"#{i + 1} {name}: Dice {dice:.4f}", fontsize=9)
                ax.axis("off")

        fig.suptitle("green = correct, red = false positive, blue = false negative", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"predictions_val_{k + 1}.png"), dpi=150)
        plt.close(fig)


