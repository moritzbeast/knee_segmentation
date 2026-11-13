# Knee X-ray Femur Segmentation

Binary segmentation of the distal femur in knee X-rays with a U-Net (`UNetLext`) in PyTorch.

- 150 grayscale X-rays (224 × 224), 60 of them with a femur mask
- Labeled images are split into 45 train / 15 validation images
- The 90 unlabeled images are used as the test set (predictions only, no metrics possible)

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12 on Apple Silicon (MPS). CUDA and CPU are selected automatically if available.

## Data

The data is not part of this repository. Expected layout:

```
data/
├── xrays/   0.png ... 149.png
└── masks/   one mask per labeled x-ray, same file name
```

X-rays without a mask are treated as test images.

## Usage

**1. Create the CSV files** (`dataset.csv`, `train.csv`, `val.csv`, `test.csv` in `data/CSVs/`):

```bash
python data_preparation.py
```

**2. Train** (final configuration):

```bash
python main.py -out_dir sessions/final -bs 8 -epochs 100 -plot_preds
```

Saves `best_model.pth` (highest validation Dice), `metrics.png` (loss and Dice curves) and, with `-plot_preds`, `predictions_val_*.png` (validation predictions before and after post-processing).

**3. Evaluate** a trained model:

```bash
python evaluate.py -out_dir sessions/final
```

Saves into the same folder:
- `val_dice.csv`: Dice per validation image, raw and post-processed
- `predictions_test_*.png`: predicted contours on all 90 test images
- `postprocessing_test_*.png`: test images changed by post-processing, before and after

Optional: `python visualize_augmentation.py` shows the effect of each augmentation on one image.

## Method

| | |
|---|---|
| Model | `UNetLext`, depth 4, base width 32, 1 input / 1 output channel |
| Loss | BCE + Dice |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Scheduler | OneCycleLR, stepped per batch |
| Training | 100 epochs, batch size 8, seed 13 |
| Normalization | mean / std computed on the training set |
| Augmentation | horizontal flip, affine (±10°, ±5 % shift, 0.9–1.1 scale), brightness / contrast ±0.2, random invert (p = 0.2) |
| Prediction | sigmoid > 0.5 |
| Post-processing | keep largest connected component, fill holes |

Image and mask are transformed together (`torchvision.transforms.v2` with `tv_tensors.Mask`), so geometric augmentations stay aligned.

## Results

Mean Dice per image on the 15 validation images (raw predictions):
The final model reaches a mean Dice of 0.9886 on the 15 validation images.“

**Augmentation** improved 12 of 15 validation images. The largest gain was on the hardest image (x-ray with markers and a bright strip at the border, 0.958 → 0.971), and the gap between training and validation curves closed.

**Post-processing** has no effect on the validation set (the model produces no islands or holes there), but it changes 16 of 90 test images, mostly by filling holes. It can also hurt: on validation image `78.png` a vertical strip splits the femur, and removing the smaller component deletes a correct part.

**Inverted test images.** 5 test images (`50`, `92`, `108`, `112`, `140`) are inverted (bone dark, background bright), while no training or validation image is. The model failed on them until `RandomInvert(p=0.2)` was added to the augmentation; now `108`, `112` and `140` are segmented correctly and `50` and `92` mostly.

## Limitations

- Only 15 validation images.
- No masks for the test set, so test results can only be checked visually.
- Training on MPS is not fully deterministic, so repeated runs give slightly different numbers.

## Files

| File | Content |
|---|---|
| `data_preparation.py` | builds the CSV files and the train/val split |
| `args.py` | command line arguments |
| `dataset.py` | `KneeDataset`, image and mask loading |
| `model.py` | `UNetLext`, taken from Aleksei Tiulpin (University of Oulu) |
| `trainer.py` | training loop, validation, plots |
| `utils.py` | Dice loss / score, mean/std, post-processing |
| `main.py` | training entry point |
| `evaluate.py` | evaluation on validation and test set |
| `visualize_augmentation.py` | visualization of the augmentations |
