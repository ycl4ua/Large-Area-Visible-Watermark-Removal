#!/usr/bin/env python3
"""Evaluate an ILAW checkpoint.

Protocols:
  fixed   : use the mask stored in the dataset (Table 1, upper section)
  coarse  : perturb the mask ``--num-coarse`` times and average (Table 1, lower
            section; the paper uses 10 random dilations)
  blind   : replace the mask with a white mask (Table 3)

Example:
  python bin/test.py \
      --config big-lama/big-lama.yaml \
      --checkpoint outputs/.../models/last.ckpt \
      --data-root /path/to/ILAW/validation \
      --protocol fixed
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.ilaw import InpaintingDataset
from src.evaluation.metrics import (
    LPIPSMetric,
    SSIMMetric,
    compute_psnr,
    compute_rmse,
    compute_rmse_masked,
)
from src.data.coarse_mask import perturb_seg
from src.training.trainers import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate ILAW watermark removal')
    p.add_argument('--config', default='big-lama/big-lama.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True,
                   help='split directory containing Watermark_free_image/, '
                        'Watermarked_image/, Mask/, Cleancomp/')
    p.add_argument('--protocol', choices=['fixed', 'coarse', 'blind'], default='fixed')
    p.add_argument('--num-coarse', type=int, default=10,
                   help='number of perturbed masks per image (coarse protocol)')
    p.add_argument('--iou-target', type=float, default=0.85,
                   help='IoU target of the perturbed masks')
    p.add_argument('--batch-size', type=int, default=1,
                   help='must be 1 for coarse/blind protocols')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--no-lpips', action='store_true')
    p.add_argument('--save-dir', default=None,
                   help='optional directory for [input | gt | pred | mask] panels')
    return p.parse_args()


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def unpad(out, batch):
    if 'unpad_to_size' not in batch:
        return out
    size = batch['unpad_to_size']
    if torch.is_tensor(size):
        if size.dim() == 2:
            h, w = int(size[0, 0]), int(size[0, 1])
        else:
            h, w = int(size[0]), int(size[1])
    else:
        h = int(size[0].reshape(-1)[0]) if torch.is_tensor(size[0]) else int(size[0])
        w = int(size[1].reshape(-1)[0]) if torch.is_tensor(size[1]) else int(size[1])
    return out[:, :, :h, :w]


def to_np(x):
    return (x.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)


def main():
    args = parse_args()
    if args.protocol in ('coarse', 'blind') and args.batch_size != 1:
        raise ValueError('--batch-size must be 1 for coarse/blind protocols')

    device = torch.device(args.device)
    config = OmegaConf.load(args.config)
    config.training_model.predict_only = True
    # The generator is fully convolutional; pad odd sizes to a multiple of 8.
    dataset = InpaintingDataset(args.data_root, img_suffix='.png', pad_out_to_modulo=8)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = load_checkpoint(config, args.checkpoint, map_location='cpu')
    model.to(device)
    model.eval()
    model.freeze()

    ssim_metric = SSIMMetric(device)
    lpips_metric = None if args.no_lpips else LPIPSMetric(device)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    results = {k: [] for k in ('psnr', 'ssim', 'rmse', 'rmsew', 'lpips')}
    seen = 0

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            gt = batch['image']
            watermarked = batch['watermarked']
            mask = batch['mask']

            if args.protocol == 'fixed':
                masks = [mask]
            elif args.protocol == 'blind':
                masks = [torch.ones_like(mask)]
            else:
                masks = []
                mask_np = (mask[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
                for _ in range(args.num_coarse):
                    cur = perturb_seg(mask_np, iou_target=args.iou_target)
                    masks.append(torch.from_numpy(cur.astype(np.float32) / 255.0)
                                 .unsqueeze(0).unsqueeze(0).to(device))

            for cur_mask in masks:
                cur_batch = dict(
                    image=gt,
                    watermarked=watermarked,
                    mask=cur_mask,
                )
                out = unpad(model(cur_batch)['inpainted'], batch)
                cur_gt = unpad(gt, batch)
                cur_mask = unpad(cur_mask, batch)

                results['psnr'].append(compute_psnr(out, cur_gt))
                results['ssim'].append(ssim_metric(out, cur_gt))
                results['rmse'].append(compute_rmse(out, cur_gt))
                results['rmsew'].append(compute_rmse_masked(out, cur_gt, cur_mask))
                if lpips_metric is not None:
                    results['lpips'].append(lpips_metric(out, cur_gt))

                if args.save_dir:
                    panel = np.concatenate([
                        to_np(watermarked[:, :, :out.shape[2], :out.shape[3]]),
                        to_np(cur_gt),
                        to_np(out),
                        np.repeat(to_np(cur_mask), 3, axis=2),
                    ], axis=1)
                    name = os.path.splitext(os.path.basename(batch['img_path'][0]))[0]
                    cv2.imwrite(os.path.join(args.save_dir, f'{name}.png'),
                                cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            seen += gt.shape[0]
            if seen % 100 == 0:
                print(f'[{seen}/{len(dataset)}] ' + ' | '.join(
                    f'{k}:{np.mean(v):.4f}' for k, v in results.items() if v))
            if args.max_images is not None and seen >= args.max_images:
                break

    print('Total:')
    for key, values in results.items():
        if values:
            print(f'  {key.upper():<6} mean={np.mean(values):.4f}  std={np.std(values):.4f}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
