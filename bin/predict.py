#!/usr/bin/env python3
"""Remove a watermark from a single image.

If ``--mask`` is omitted, a white mask is used (the paper's blind setting).

Example:
  python bin/predict.py \
      --config big-lama/big-lama.yaml \
      --checkpoint outputs/.../models/last.ckpt \
      --image input.png --mask mask.png --out restored.png
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.training.trainers import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description='Single-image watermark removal')
    p.add_argument('--config', default='big-lama/big-lama.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--mask', default=None,
                   help='coarse watermark mask; omit for the blind (white-mask) setting')
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_rgb(path):
    img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


def load_mask(path, shape):
    if path is None:
        return np.ones((1, shape[0], shape[1]), dtype=np.float32)
    mask = np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0
    return (mask > 0.5).astype(np.float32)[None, ...]


def pad_to_modulo(arr, mod=8):
    c, h, w = arr.shape
    ph, pw = (-h) % mod, (-w) % mod
    return np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode='symmetric'), h, w


def main():
    args = parse_args()
    device = torch.device(args.device)

    config = OmegaConf.load(args.config)
    config.training_model.predict_only = True
    model = load_checkpoint(config, args.checkpoint, map_location='cpu')
    model.to(device).eval().freeze()

    watermarked = load_rgb(args.image)
    mask = load_mask(args.mask, watermarked.shape[1:])
    watermarked_pad, h, w = pad_to_modulo(watermarked)
    mask_pad, _, _ = pad_to_modulo(mask)

    batch = {
        'image': torch.from_numpy(watermarked_pad).unsqueeze(0).to(device),
        'watermarked': torch.from_numpy(watermarked_pad).unsqueeze(0).to(device),
        'mask': torch.from_numpy(mask_pad).unsqueeze(0).to(device),
    }
    with torch.no_grad():
        out = model(batch)['inpainted'][:, :, :h, :w]
    out = (out[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

    out_path = args.out or args.image.rsplit('.', 1)[0] + '_restored.png'
    Image.fromarray(out).save(out_path)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
