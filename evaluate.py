"""evaluate.py

Evaluation utilities used to measure a trained UNet model on the test split.

The script loads `best_model.weights.h5` from the checkpoint directory,
runs the model on the test dataset and computes common saliency metrics:
IoU, Precision, Recall, F1 and MAE. It also saves visual examples (input,
ground-truth, predicted mask and overlay) to help qualitative analysis.

Typical usage is to point `--checkpoint-dir` to the folder created by
`train.py` and set an output directory for the sample figures.
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sod_model import get_sod_model
import data_loader


def soft_iou_numpy(y_true, y_pred, eps=1e-7):
    y_true_f = y_true.reshape(y_true.shape[0], -1)
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1)
    intersection = (y_true_f * y_pred_f).sum(axis=1)
    union = (y_true_f + y_pred_f - y_true_f * y_pred_f).sum(axis=1)
    iou = (intersection + eps) / (union + eps)
    return iou


def evaluate(args):
    # Load the CSV that lists test image and mask paths. Fail early if the
    # split is missing so the caller can regenerate splits if needed.
    test_csv = os.path.join(args.splits_dir, 'test.csv')
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f'Missing test split at {test_csv}')

    df_test = pd.read_csv(test_csv)
    test_ds = data_loader.create_dataset(df_test['image'].tolist(), df_test['mask'].tolist(), batch_size=args.batch_size, target_size=args.target_size, shuffle=False, augment=False)

    # Build model matching training-time configuration and load the saved
    # best weights. We only load weights (not optimizer state) since this is
    # an evaluation-only run.
    model = get_sod_model(input_shape=(args.target_size, args.target_size, 3), base_filters=args.base_filters, use_batchnorm=args.batchnorm, use_dropout=args.dropout, dropout_rate=args.dropout_rate)
    best_path = os.path.join(args.checkpoint_dir, 'best_model.weights.h5')
    if not os.path.exists(best_path):
        raise FileNotFoundError(f'Best weights not found at {best_path}')
    model.load_weights(best_path)
    print('Loaded weights from', best_path)

    all_ious = []
    all_prec = []
    all_rec = []
    all_f1 = []
    all_mae = []

    os.makedirs(args.out_dir, exist_ok=True)

    sample_count = 0
    for images, masks in test_ds:
        preds = model(images, training=False).numpy()
        gt = masks.numpy()

        # Soft IoU compares binarized ground-truth with predicted
        # probabilities (no thresholding) and produces a per-image IoU.
        ious = soft_iou_numpy((gt > 0.5).astype(np.float32), preds.astype(np.float32))
        all_ious.extend(ious.tolist())

        # Binary masks used for precision/recall/F1: threshold the
        # predictions using the CLI `--threshold` argument.
        preds_bin = (preds >= args.threshold).astype(np.float32)
        gt_bin = (gt >= 0.5).astype(np.float32)

        # per-image precision/recall/f1
        for p, g in zip(preds_bin, gt_bin):
            p_f = p.reshape(-1)
            g_f = g.reshape(-1)
            tp = (p_f * g_f).sum()
            fp = ((p_f == 1) & (g_f == 0)).sum()
            fn = ((p_f == 0) & (g_f == 1)).sum()
            prec = tp / (tp + fp + 1e-7)
            rec = tp / (tp + fn + 1e-7)
            f1 = 2 * prec * rec / (prec + rec + 1e-7)
            mae = np.mean(np.abs(p.reshape(-1) - g.reshape(-1)))
            all_prec.append(float(prec))
            all_rec.append(float(rec))
            all_f1.append(float(f1))
            all_mae.append(float(mae))

        # save sample visualizations
        if sample_count < args.num_samples:
            bs = images.shape[0]
            for i in range(bs):
                if sample_count >= args.num_samples:
                    break
                inp = images[i].numpy()
                gt_mask = gt[i].squeeze()
                pred_mask = preds[i].squeeze()

                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                axes[0].imshow(inp)
                axes[0].set_title('Input')
                axes[0].axis('off')

                axes[1].imshow(gt_mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(pred_mask, cmap='gray')
                axes[2].set_title('Predicted')
                axes[2].axis('off')

                # overlay: red mask on image
                overlay = inp.copy()
                mask_col = np.stack([pred_mask, np.zeros_like(pred_mask), np.zeros_like(pred_mask)], axis=-1)
                overlay = (overlay * 255).astype(np.uint8)
                alpha = np.clip(pred_mask[..., None], 0, 1) * 0.6
                overlay = (overlay * (1 - alpha) + (mask_col * 255) * alpha).astype(np.uint8)
                axes[3].imshow(overlay)
                axes[3].set_title('Overlay')
                axes[3].axis('off')

                out_path = os.path.join(args.out_dir, f'sample_{sample_count:03d}.png')
                fig.tight_layout()
                fig.savefig(out_path)
                plt.close(fig)
                sample_count += 1

    # Helper to compute mean while avoiding empty-list errors.
    def mean(x):
        return float(np.mean(x)) if len(x) else float('nan')

    results = {
        'IoU': mean(all_ious),
        'Precision': mean(all_prec),
        'Recall': mean(all_rec),
        'F1': mean(all_f1),
        'MAE': mean(all_mae),
    }

    print('Evaluation results:')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')

    # save results json
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        import json

        json.dump(results, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits-dir', type=str, default='data/splits')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--target-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--out-dir', type=str, default='eval_outputs')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--base-filters', type=int, default=64)
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
