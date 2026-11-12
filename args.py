import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Knee Segmentation project")

    parser.add_argument("-csv_dir", default="data/CSVs")

    parser.add_argument("-out_dir", default="sessions")

    parser.add_argument("-epochs", type=int, default=100)

    parser.add_argument("-bs", type=int, default=16)

    parser.add_argument("-lr", type=float, default=1e-3)

    parser.add_argument("-wd", type=float, default=1e-4)

    parser.add_argument("-seed", type=int, default=13)

    parser.add_argument("-depth", type=int, default=4)

    parser.add_argument("-plot_preds", action="store_true")

    args = parser.parse_args()
    
    return args