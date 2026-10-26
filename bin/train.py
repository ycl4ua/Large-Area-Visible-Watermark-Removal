#!/usr/bin/env python3

import logging
import os
import sys
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.training.trainers import load_checkpoint, make_training_model
from src.utils import register_debug_signal_handlers, handle_ddp_subprocess, handle_ddp_parent_process, \
    handle_deterministic_config

LOGGER = logging.getLogger(__name__)


@handle_ddp_subprocess()
@hydra.main(config_path='../big-lama', config_name='big-lama')
def main(config: OmegaConf):
    try:
        need_set_deterministic = handle_deterministic_config(config)

        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        is_in_ddp_subprocess = handle_ddp_parent_process()

        if not is_in_ddp_subprocess:
            LOGGER.info(OmegaConf.to_yaml(config))
            OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml'))

        checkpoints_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(checkpoints_dir, exist_ok=True)

        # there is no need to suppress this logger in ddp, because it handles rank on its own
        metrics_logger = TensorBoardLogger(to_absolute_path(config.location.tb_dir),
                                           name=os.path.basename(os.getcwd()))
        metrics_logger.log_hyperparams(config)

        pretrained_path = config.location.get('pretrained_model_path', None)
        if pretrained_path:
            checkpoint_path = to_absolute_path(pretrained_path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f'Big-LaMa checkpoint not found: {checkpoint_path}. '
                    'Download it or unset location.pretrained_model_path.'
                )
            LOGGER.info(f'Initialize generator from Big-LaMa: {checkpoint_path}')
            training_model = load_checkpoint(config, checkpoint_path, strict=False, map_location='cpu')
        else:
            LOGGER.info('Train from random initialization (no Big-LaMa checkpoint).')
            training_model = make_training_model(config)

        trainer_kwargs = OmegaConf.to_container(config.trainer.kwargs, resolve=True)
        if need_set_deterministic:
            trainer_kwargs['deterministic'] = True

        trainer = Trainer(
            # there is no need to suppress checkpointing in ddp, because it handles rank on its own
            callbacks=ModelCheckpoint(dirpath=checkpoints_dir, **config.trainer.checkpoint_kwargs),
            logger=metrics_logger,
            default_root_dir=os.getcwd(),
            **trainer_kwargs
        )
        trainer.fit(training_model)
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Training failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
