from torch.utils.data import Dataset
import cv2
import numpy as np

class KneeDataset(Dataset):
    def __init__(self, df, mean, std, transform=None):
        self.df = df
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        xray = read_xray(self.df["xrays"].iloc[index])
        mask = read_mask(self.df["masks"].iloc[index])

        if self.transform is not None:
            for t in self.transform:
                xray = t(xray)
                mask = t(mask)

        xray = (xray - self.mean) / self.std

        res = {
            "xray" : xray,
            "mask" : mask
        }

        return res
        
def read_xray(dir):
    xray = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    xray = xray.astype(np.float32) / 255
    xray = xray.reshape(1, *xray.shape) # 1 H W

    return xray

def read_mask(dir):
    mask = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.float32)
    mask = mask.reshape(1, *mask.shape)

    return mask