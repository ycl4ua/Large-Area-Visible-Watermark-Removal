import logging

import torch
from omegaconf import OmegaConf

from src.training.losses.feature_matching import feature_matching_loss, masked_l1_loss
from src.training.trainers.base import BaseInpaintingTrainingModule
from src.utils import add_prefix_to_keys

LOGGER = logging.getLogger(__name__)


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    """Dual-branch training module for ILAW.

    The generator receives two inputs:
      * ``masked_img``  : watermarked * (1 - mask), concatenated with the mask
      * ``watermarked`` : the watermarked image, concatenated with the mask

    It returns the restored image ``Y`` and the background component ``C_bkg``.
    """

    def __init__(self, *args, concat_mask=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_mask = concat_mask

    def forward(self, batch):
        mask = batch['mask'].float()
        watermarked = batch['watermarked']

        masked_img = watermarked * (1 - mask)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)
            watermarked_input = torch.cat([watermarked, mask], dim=1)
        else:
            watermarked_input = watermarked

        batch['predicted_image'], batch['predicted_comp'] = self.generator(masked_img, watermarked_input)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['watermarked']
        return batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch['predicted_image']
        mask = batch['mask']

        # Eq. (6): L1 on the final image Y and on C_bkg.
        l1_value = masked_l1_loss(predicted_img, img, mask,
                                  self.config.losses.l1.weight_known,
                                  self.config.losses.l1.weight_missing)
        l1_value = l1_value + masked_l1_loss(batch['predicted_comp'], batch['comp'], mask,
                                             self.config.losses.l1.weight_known,
                                             self.config.losses.l1.weight_missing)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        # Patch-wise adversarial loss (Eq. 8/9).
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(
            real_batch=img, fake_batch=predicted_img,
            discr_real_pred=discr_real_pred, discr_fake_pred=discr_fake_pred,
            mask=mask)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # Discriminator feature matching (Eq. 10, lambda_4=100).
        if self.config.losses.feature_matching.weight > 0:
            need_mask_in_fm = OmegaConf.to_container(self.config.losses.feature_matching).get('pass_mask', False)
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask if need_mask_in_fm else None
                                             ) * self.config.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

        # ResNet50 perceptual loss on Y and C_bkg (Eq. 7).
        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value

        return total_loss, metrics

    def discriminator_loss(self, batch):
        predicted_img = batch['predicted_image'].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, _ = self.discriminator(batch['image'])
        discr_fake_pred, _ = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(
            real_batch=batch['image'], fake_batch=predicted_img,
            discr_real_pred=discr_real_pred, discr_fake_pred=discr_fake_pred,
            mask=batch['mask'])
        return adv_discr_loss, dict(discr_adv=adv_discr_loss, **add_prefix_to_keys(adv_metrics, 'adv_'))
