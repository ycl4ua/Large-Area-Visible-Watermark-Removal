import logging

from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from src.data.ilaw import InpaintingDataset

LOGGER = logging.getLogger(__name__)


def make_default_train_dataloader(indir, kind='ilaw', img_suffix='.png',
                                  dataloader_kwargs=None, ddp_kwargs=None, **kwargs):
    """Build the ILAW training dataloader."""
    if kind not in ('ilaw', 'default'):
        raise ValueError(f'Unknown train dataset kind {kind}')

    LOGGER.info(f'Make train dataloader for ILAW from {indir}')
    dataset = InpaintingDataset(indir, img_suffix=img_suffix, **kwargs)

    dataloader_kwargs = dict(dataloader_kwargs or {})
    if ddp_kwargs is not None:
        dataloader_kwargs['shuffle'] = False
        dataloader_kwargs['sampler'] = DistributedSampler(dataset, **ddp_kwargs)
    return DataLoader(dataset, **dataloader_kwargs)


def make_default_val_dataset(indir, kind='ilaw', img_suffix='.png', **kwargs):
    """Build the ILAW validation/evaluation dataset."""
    if OmegaConf.is_list(indir) or isinstance(indir, (tuple, list)):
        return ConcatDataset([
            make_default_val_dataset(cur_indir, kind=kind, img_suffix=img_suffix, **kwargs)
            for cur_indir in indir
        ])

    if kind not in ('ilaw', 'default'):
        raise ValueError(f'Unknown val dataset kind {kind}')
    LOGGER.info(f'Make val dataset for ILAW from {indir}')
    return InpaintingDataset(indir, img_suffix=img_suffix, **kwargs)


def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)
    return DataLoader(dataset, **(dataloader_kwargs or {}))
