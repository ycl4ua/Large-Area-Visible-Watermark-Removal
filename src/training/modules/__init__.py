import logging

from src.training.modules.ffc import FFCResNetGenerator
from src.training.modules.pix2pixhd import NLayerDiscriminator


def make_generator(config, kind, **kwargs):
    logging.info(f'Make generator {kind}')
    if kind == 'ffc_resnet':
        return FFCResNetGenerator(**kwargs)
    raise ValueError(f'Unknown generator kind {kind}')


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')
    if kind == 'pix2pixhd_nlayer':
        return NLayerDiscriminator(**kwargs)
    raise ValueError(f'Unknown discriminator kind {kind}')
