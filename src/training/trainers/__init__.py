import logging

from src.training.trainers.checkpoint import (
    is_biglama_state_dict,
    load_biglama_weights,
    torch_load,
)
from src.training.trainers.default import DefaultInpaintingTrainingModule

LOGGER = logging.getLogger(__name__)


def get_training_model_class(kind):
    if kind == 'default':
        return DefaultInpaintingTrainingModule
    raise ValueError(f'Unknown trainer module {kind}')


def make_training_model(config):
    kind = config.training_model.kind
    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'

    LOGGER.info(f'Make training model {kind}')
    return get_training_model_class(kind)(config, **kwargs)


def load_checkpoint(train_config, path, map_location='cuda', strict=False):
    """Build the training model and load weights from ``path``.

    Two kinds of checkpoints are supported:
      * ILAW / Lightning checkpoints (keys like ``generator.model_init.*``)
      * the original Big-LaMa checkpoint (keys like ``generator.model.*``)
    """
    model = make_training_model(train_config)
    state = torch_load(path, map_location)
    state_dict = state.get('state_dict', state) if isinstance(state, dict) else state

    if is_biglama_state_dict(state_dict):
        load_biglama_weights(model, state_dict)
    else:
        incompatible = model.load_state_dict(state_dict, strict=strict)
        # A predict-only model has no discriminator/loss modules; do not warn
        # about their keys when loading a training checkpoint for evaluation.
        ignored_prefixes = ('discriminator.', 'adversarial_loss.', 'loss_', 'val_evaluator.')
        missing = [k for k in incompatible.missing_keys if not k.startswith(ignored_prefixes)]
        unexpected = [k for k in incompatible.unexpected_keys if not k.startswith(ignored_prefixes)]
        if missing:
            LOGGER.warning('Missing keys: %s', missing[:10])
        if unexpected:
            LOGGER.warning('Unexpected keys: %s', unexpected[:10])

    model.on_load_checkpoint(state if isinstance(state, dict) else {'state_dict': state_dict})
    return model
