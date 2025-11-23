# Salient Object Detection (SOD)

Minimal Salient Object Detection pipeline implemented with TensorFlow/Keras.
This repository contains a configurable UNet implementation, a custom
training loop with checkpointing and mixed-precision support, a dataset
preparer, an evaluation utility, and a small demo notebook for inference.

**Project layout (important files)**
- `sod_model.py`: configurable UNet model builder (`get_sod_model`).
- `data_loader.py`: builds CSV splits and creates `tf.data.Dataset` with
	augmentations used for training/validation/test.
- `train.py`: training script with custom loop, BCE+Dice loss, LR scheduling,
	early stopping and checkpoint saving (`best_model.weights.h5`).
- `evaluate.py`: runs evaluation on a checkpoint and saves metrics + figures.
- `demos/demo_inference.ipynb`: interactive notebook for quick inference
	and visualization.

Quick start
-----------
Prerequisites: Python 3.10+ (project was developed with a `tf-gpu-env`
virtualenv provided in the repo). Activate your environment before running
commands below.

```bash
source tf-gpu-env/bin/activate
```

1) Prepare dataset splits (example for DUTS-TR):

```bash
python data_loader.py --dataset-root data/DUTS-TR/DUTS-TR --target-size 224 --max-samples 2500 --quick-check
```

This writes `data/splits/{train.csv, val.csv, test.csv}` containing paired
image and mask file paths.

2) Train a UNet model (example):

```bash
python train.py \
	--splits-dir data/splits \
	--target-size 224 \
	--batch-size 4 \
	--epochs 50 \
	--checkpoint-dir checkpoints_unet_experiment \
	--augment --dropout \
	--base-filters 32 --num-blocks 5 \
	--lr 1e-4 --label-smoothing 0.02 --mixed-precision
```

Notes:
- Reduce `--batch-size` or `--base-filters` if you hit GPU OOMs.
- Use `--mixed-precision` to reduce memory and speed up training on
	supported GPUs.

3) Evaluate a saved checkpoint:

```bash
python evaluate.py \
	--splits-dir data/splits \
	--checkpoint-dir checkpoints_unet_experiment \
	--target-size 224 \
	--batch-size 8 \
	--out-dir eval_outputs --num-samples 50
```

This computes IoU, Precision, Recall, F1, MAE and saves example visual
overlays to `eval_outputs/` and `results.json`.

Demo / quick inference
----------------------
Open the demo notebook to run inference interactively and generate
visual overlays:

```bash
jupyter notebook demos/demo_inference.ipynb
```

Reproducibility & checkpoints
-----------------------------
- Best model weights are saved as `best_model.weights.h5` inside the
	folder you pass to `--checkpoint-dir` in `train.py`.
- Common checkpoint folders used in experiments: `checkpoints_exp7_DEEP`,
	`checkpoints_unet64_5b_full`, etc. Inspect those directories for
	`best_model.weights.h5` and training logs.

Tips for improving results
-------------------------
- Stronger augmentations and label smoothing helped avoid overfitting in
	our experiments.
- If the target is higher F1, try:
	- Increasing training epochs and enabling `--augment`.
	- Tuning loss weighting (increase Dice weight) or using focal loss.
	- Ensembling multiple runs or test-time augmentation (TTA).

License / Attribution
---------------------
This repository is a teaching/example project. Add a license file before
publishing to GitHub (e.g. `MIT` or `Apache-2.0`) if you want to make the
code public.

Questions or next steps
-----------------------
If you'd like, I can:
- Add a small `CONTRIBUTING.md` with exact commands to reproduce an
	experiment (training + evaluation).
- Add a `requirements.txt` file listing the versions used to run the
	experiments.

