import glob
import os

import cv2
import PIL.Image as Image
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


class InpaintingDataset(Dataset):
    """ILAW reader.

    Expected layout::

        <datadir>/
            Watermark_free_image/   Gwf
            Watermarked_image/      X
            Mask/                   M
            Cleancomp/              Gbkg
            alpha/                  (optional) A

    Folder names are case-sensitive.  The first four folders must contain the
    same file stems.  alpha is only required when load_alpha=True (used by the
    coarse-mask evaluation).
    """

    _IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

    def __init__(self, datadir, img_suffix='.png', pad_out_to_modulo=None,
                 scale_factor=None, load_alpha=False, **unused):
        self.datadir = datadir
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor
        self.load_alpha = load_alpha

        self.gt_dir = self._resolve_dir('Watermark_free_image')
        self.watermarked_dir = self._resolve_dir('Watermarked_image')
        self.mask_dir = self._resolve_dir('Mask')
        self.comp_dir = self._resolve_dir('Cleancomp')
        self.alpha_dir = self._resolve_dir('alpha') if load_alpha else None

        self.gt_files = self._index(self.gt_dir, img_suffix, key='Watermark_free_image')
        self.watermarked_files = self._index(self.watermarked_dir, img_suffix, key='Watermarked_image')
        self.mask_files = self._index(self.mask_dir, None, key='Mask')
        self.comp_files = self._index(self.comp_dir, img_suffix, key='Cleancomp')
        self.alpha_files = self._index(self.alpha_dir, None, key='alpha') if self.alpha_dir else {}

        self.stems = sorted(self.gt_files)
        if not self.stems:
            raise FileNotFoundError(f'No images found in {self.gt_dir}')
        for name, files in (('Watermarked_image', self.watermarked_files),
                            ('Mask', self.mask_files),
                            ('Cleancomp', self.comp_files)):
            self._check_pairing(name, files)
        if self.alpha_dir:
            self._check_pairing('alpha', self.alpha_files)

    def _resolve_dir(self, name):
        path = os.path.join(self.datadir, name)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f'Cannot find {name}/ under {self.datadir}. Expected folder names: '
                'Watermark_free_image, Watermarked_image, Mask, Cleancomp '
                '(and alpha/ when load_alpha=True).')
        return path

    @classmethod
    def _index(cls, directory, suffix, key):
        files = {}
        for fname in sorted(glob.glob(os.path.join(directory, '*'))):
            if not fname.lower().endswith(cls._IMAGE_EXT):
                continue
            if suffix is not None and not fname.lower().endswith(suffix.lower()):
                continue
            files[os.path.splitext(os.path.basename(fname))[0]] = fname
        if not files:
            raise FileNotFoundError(f'No {key} images found in {directory}')
        return files

    def _check_pairing(self, name, files):
        missing = set(self.stems) - set(files)
        extra = set(files) - set(self.stems)
        if missing or extra:
            raise ValueError(
                f'{name} folder does not match the watermark-free images. '
                f'Missing: {sorted(missing)[:5]}..., extra: {sorted(extra)[:5]}...')

    def __len__(self):
        return len(self.stems)

    def _load_and_augment(self, result):
        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['watermarked'] = scale_image(result['watermarked'], self.scale_factor)
            result['comp'] = scale_image(result['comp'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor,
                                         interpolation=cv2.INTER_NEAREST)
            if 'alpha' in result:
                result['alpha'] = scale_image(result['alpha'], self.scale_factor,
                                              interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            for key in ('image', 'watermarked', 'comp', 'mask', 'alpha'):
                if key in result:
                    result[key] = pad_img_to_modulo(result[key], self.pad_out_to_modulo)
        return result

    def __getitem__(self, i):
        stem = self.stems[i]
        result = dict(
            image=load_image(self.gt_files[stem], mode='RGB'),
            watermarked=load_image(self.watermarked_files[stem], mode='RGB'),
            comp=load_image(self.comp_files[stem], mode='RGB'),
            mask=load_image(self.mask_files[stem], mode='L')[None, ...],
        )
        if self.alpha_dir:
            result['alpha'] = load_image(self.alpha_files[stem], mode='L')[None, ...]
        result['img_path'] = self.gt_files[stem]
        return self._load_and_augment(result)
