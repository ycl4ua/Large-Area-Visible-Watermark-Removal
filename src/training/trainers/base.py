import logging
from typing import Dict, Tuple

import pandas as pd
import pytorch_lightning as ptl
import torch

from src.evaluation import make_evaluator
from src.data.datasets import make_default_train_dataloader, make_default_val_dataloader
from src.training.losses.adversarial import make_discrim_loss
from src.training.losses.perceptual import ResNetPL
from src.training.modules import make_generator, make_discriminator
from src.utils import add_prefix_to_keys, average_dicts, set_requires_grad, flatten_dict, \
    get_has_ddp_rank

LOGGER = logging.getLogger(__name__)


def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


class BaseInpaintingTrainingModule(ptl.LightningModule):
    def __init__(self, config, use_ddp, predict_only=False, **kwargs):
        super().__init__(**kwargs)
        LOGGER.info('BaseInpaintingTrainingModule init called')

        self.config = config
        self.generator = make_generator(config, **self.config.generator)
        self.use_ddp = use_ddp
        self.loss_resnet_pl = None

        if not get_has_ddp_rank():
            LOGGER.info(f'Generator\n{self.generator}')

        if not predict_only:
            self.save_hyperparameters(self.config)
            self.discriminator = make_discriminator(**self.config.discriminator)
            self.adversarial_loss = make_discrim_loss(**self.config.losses.adversarial)
            self.val_evaluator = make_evaluator(**self.config.evaluator)

            if not get_has_ddp_rank():
                LOGGER.info(f'Discriminator\n{self.discriminator}')

            if self.config.losses.get('resnet_pl', {'weight': 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL(**self.config.losses.resnet_pl)

        LOGGER.info('BaseInpaintingTrainingModule init done')

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            dict(optimizer=make_optimizer(self.generator.parameters(), **self.config.optimizers.generator)),
            dict(optimizer=make_optimizer(discriminator_params, **self.config.optimizers.discriminator)),
        ]

    def train_dataloader(self):
        kwargs = dict(self.config.data.train)
        if self.use_ddp:
            kwargs['ddp_kwargs'] = dict(num_replicas=self.get_world_size(),
                                        rank=self.trainer.global_rank,
                                        shuffle=True)
        return make_default_train_dataloader(**kwargs)

    def val_dataloader(self):
        return [make_default_val_dataloader(**self.config.data.val)]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self._do_step(batch, batch_idx, mode='train', optimizer_idx=optimizer_idx)

    def validation_step(self, batch, batch_idx):
        return self._do_step(batch, batch_idx, mode='val')

    def training_step_end(self, batch_parts_outputs):
        full_loss = (batch_parts_outputs['loss'].mean()
                     if torch.is_tensor(batch_parts_outputs['loss'])
                     else torch.tensor(batch_parts_outputs['loss']).float().requires_grad_(True))
        log_info = {k: v.mean() for k, v in batch_parts_outputs['log_info'].items()}
        self.log_dict(log_info, on_step=True, on_epoch=False)
        return full_loss

    def validation_epoch_end(self, outputs):
        averaged_logs = average_dicts(step_out['log_info'] for step_out in outputs)
        self.log_dict({k: v.mean() for k, v in averaged_logs.items()})

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        val_evaluator_states = [s['val_evaluator_state'] for s in outputs if 'val_evaluator_state' in s]
        val_evaluator_res = self.val_evaluator.evaluation_end(states=val_evaluator_states)
        val_evaluator_res_df = pd.DataFrame(val_evaluator_res).stack(1).unstack(0)
        val_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
        LOGGER.info(f'Validation metrics after epoch #{self.current_epoch}, '
                    f'total {self.global_step} iterations:\n{val_evaluator_res_df}')

        for k, v in flatten_dict(val_evaluator_res).items():
            self.log(f'val_{k}', v)

    def _do_step(self, batch, batch_idx, mode='train', optimizer_idx=None):
        if optimizer_idx == 0:  # generator
            set_requires_grad(self.generator, True)
            set_requires_grad(self.discriminator, False)
        elif optimizer_idx == 1:  # discriminator
            set_requires_grad(self.generator, False)
            set_requires_grad(self.discriminator, True)

        batch = self(batch)

        total_loss = 0
        metrics = {}

        if optimizer_idx is None or optimizer_idx == 0:
            total_loss, metrics = self.generator_loss(batch)
        elif optimizer_idx is None or optimizer_idx == 1:
            if self.config.losses.adversarial.weight > 0:
                total_loss, metrics = self.discriminator_loss(batch)

        result = dict(loss=total_loss, log_info=add_prefix_to_keys(metrics, f'{mode}_'))
        if mode == 'val':
            result['val_evaluator_state'] = self.val_evaluator.process_batch(batch)
        return result

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the generator and add ``predicted_image``, ``predicted_comp`` and
        ``inpainted`` to the batch."""
        raise NotImplementedError()

    def generator_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def discriminator_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def get_world_size(self):
        # PL 1.x renamed num_processes -> world_size in later releases.
        world_size = getattr(self.trainer, 'world_size', None)
        if world_size is None:
            world_size = getattr(self.trainer, 'num_nodes', 1) * getattr(self.trainer, 'num_processes', 1)
        return world_size
