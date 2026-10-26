"""Checkpoint helpers for initializing the ILAW generator from Big-LaMa."""
import logging

import torch

LOGGER = logging.getLogger(__name__)


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # torch < 1.13 has no weights_only argument
        return torch.load(path, map_location=map_location)


def is_biglama_state_dict(state_dict):
    return any(key.startswith('generator.model.') for key in state_dict)


def remap_biglama_key(key):
    """Map ``generator.model.<idx>.*`` of the original LaMa generator to the
    split ILAW generator (model_init / model1 / model2 / model3 / model_final).

    The split exists because the ILAW generator exposes the three intermediate
    feature maps consumed by the GFM modules.
    """
    parts = key.split('.')
    idx = int(parts[2])
    rest = '.'.join(parts[3:])

    if 0 <= idx <= 4:
        return f'generator.model_init.{idx}.{rest}'
    if 5 <= idx <= 10:
        return f'generator.model1.{idx - 5}.{rest}'
    if 11 <= idx <= 16:
        return f'generator.model2.{idx - 11}.{rest}'
    if 17 <= idx <= 22:
        return f'generator.model3.{idx - 17}.{rest}'
    if 23 <= idx <= 36:
        return f'generator.model_final.{idx - 23}.{rest}'
    return None


def copy_center(src, dst):
    """Copy overlapping regions of two tensors with different shapes."""
    if src.dim() != dst.dim():
        return False
    slices_dst = []
    slices_src = []
    for d in range(src.dim()):
        n = min(src.shape[d], dst.shape[d])
        if d in (0, 1):
            off_dst = off_src = 0
        else:
            off_dst = (dst.shape[d] - n) // 2
            off_src = (src.shape[d] - n) // 2
        slices_dst.append(slice(off_dst, off_dst + n))
        slices_src.append(slice(off_src, off_src + n))
    dst[tuple(slices_dst)] = src[tuple(slices_src)]
    return True


AUX_PREFIXES = ('val_evaluator.', 'test_evaluator.', 'loss_', 'adversarial_loss.')


def load_biglama_weights(model, state_dict):
    """Initialize the ILAW generator from a Big-LaMa checkpoint.

    Model weights live under ``generator.model.*``; the new WCC/BCE/GFM modules
    are intentionally left random.  Evaluator/loss buffers are skipped because
    they are re-created from the training config.  Unexpected model keys are
    reported instead of being silently ignored.
    """
    dst_state_dict = model.state_dict()
    loaded, mismatched, aux_skipped, missing, unexpected = 0, [], [], [], []

    for src_name, src_param in state_dict.items():
        if src_name.startswith('generator.model.'):
            new_name = remap_biglama_key(src_name)
        elif src_name.startswith('discriminator.'):
            new_name = src_name
        elif src_name.startswith(AUX_PREFIXES):
            aux_skipped.append(src_name)
            continue
        else:
            unexpected.append(src_name)
            continue

        if new_name is None or new_name not in dst_state_dict:
            missing.append(src_name)
            continue
        dst_param = dst_state_dict[new_name]
        if src_param.shape != dst_param.shape:
            LOGGER.warning('Size mismatch for %s: %s -> %s', new_name,
                           tuple(src_param.shape), tuple(dst_param.shape))
            if copy_center(src_param.data, dst_param):
                mismatched.append(new_name)
            continue
        dst_param.copy_(src_param.data)
        loaded += 1

    LOGGER.info('Big-LaMa init: loaded %d tensors, %d size-mismatched, '
                '%d auxiliary keys skipped', loaded, len(mismatched), len(aux_skipped))
    if missing:
        LOGGER.warning('Big-LaMa keys not found in the ILAW model: %s', missing[:10])
    if unexpected:
        LOGGER.warning('Unexpected Big-LaMa keys not loaded: %s', unexpected[:10])
    return loaded
