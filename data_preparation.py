import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def create_dataset_csv(csv_dir="data/CSVs", xray_dir="data/xrays", mask_dir="data/masks"):
    os.makedirs(csv_dir, exist_ok=True)

    data = []

    # sorted so the csv is the same on every device
    for xray in sorted(os.listdir(xray_dir)):
        if not xray.endswith(".png"):
            continue

        xray_path = os.path.join(xray_dir, xray)
        mask_path = os.path.join(mask_dir, xray)

        # xrays without a mask are test
        if not os.path.exists(mask_path):
            mask_path = None

        data.append((xray_path, mask_path))

    df = pd.DataFrame(data, columns=["xrays", "masks"])

    # na_rep writes None instead of empty cell
    df.to_csv(os.path.join(csv_dir, "dataset.csv"), index=False, na_rep="None")

    return df


def create_split_csvs(csv_dir="data/CSVs", val_size=0.25, seed=13):
    df = pd.read_csv(os.path.join(csv_dir, "dataset.csv"))

    # images with mask None are the test set
    test_df = df[df["masks"].isna()]
    labeled_df = df[df["masks"].notna()]

    # split the labeled images into train and val (75/25)
    train_df, val_df = train_test_split(labeled_df, test_size=val_size, random_state=seed)

    train_df.to_csv(os.path.join(csv_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(csv_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(csv_dir, "test.csv"), index=False, na_rep="None")

    return train_df, val_df, test_df


def plot_csv_sizes(csv_dir="data/CSVs", out_dir="data/outputs"):
    os.makedirs(out_dir, exist_ok=True)

    names = ["dataset", "train", "val", "test"]

    counts = [len(pd.read_csv(os.path.join(csv_dir, f"{name}.csv"))) for name in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, counts, width=0.6, color="#3b6fd8")
    ax.bar_label(bars, padding=3)

    ax.set_title("Number of samples per CSV file")
    ax.set_ylabel("Samples")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "csv_sizes.png"), dpi=150)
    plt.close(fig)

    return dict(zip(names, counts))


if __name__ == "__main__":
    from args import get_args

    args = get_args()

    create_dataset_csv(csv_dir=args.csv_dir)
    create_split_csvs(csv_dir=args.csv_dir)
    print(plot_csv_sizes(csv_dir=args.csv_dir))
