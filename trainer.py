from args import get_args
import torch
import torch.nn as nn
from utils import dice_loss_from_logits, dice_score_from_logits

args = get_args()
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def train_model(model, train_loader, val_loader):
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0

        all_labels = []
        all_probs = []

        for batch in train_loader:
            images = batch["xray"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        validate_model(model, val_loader, bce, device)




def validate_model(model, val_loader, bce, device):

    model.eval()
    running_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["xray"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            running_loss += loss.item()

        val_loss = running_loss / len(val_loader)

        return val_loss