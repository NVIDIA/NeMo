# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import ceil, floor
from multiprocessing.sharedctypes import Value
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from src.audio_to_multi_label import get_audio_multi_label_dataset, get_tarred_audio_multi_label_dataset
from torchmetrics import Accuracy

from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import *
from nemo.utils import logging, model_utils


class SigmoidFocalLoss(torch.nn.Module):
    """
    Adapted from https://pytorch.org/vision/0.12/_modules/torchvision/ops/focal_loss.html
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean") -> None:
        """
        Args:
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor with the reduction option applied.
        """

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class EncDecMultiClassificationModel(EncDecClassificationModel):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"outputs": NeuralType(('B', 'T', 'C'), LogitsType())}

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        if cfg.get("is_regression_task", False):
            raise ValueError(f"EndDecClassificationModel requires the flag is_regression_task to be set as false")
        self.num_classes = len(cfg.labels)
        self.eval_loop_cnt = 0
        super().__init__(cfg=cfg, trainer=trainer)

    @classmethod
    def list_available_models(cls) -> Optional[List[PretrainedModelInfo]]:
        return []

    def _setup_metrics(self):
        self._accuracy = TopKClassificationAccuracy(dist_sync_on_step=True)
        self._macro_accuracy = Accuracy(num_classes=self.num_classes, average='macro')

    def _setup_loss(self):
        if "loss" in self.cfg:
            if "type" in self.cfg.loss and "focal" in self.cfg.loss.type:
                alpha = self.cfg.loss.get("alpha", 0.25)
                gamma = self.cfg.loss.get("gamma", 2.0)
                logging.info(f"Using focal loss with alpha={alpha} and gamma={gamma}")
                return SigmoidFocalLoss(alpha, gamma)
            weight = self.cfg.loss.get("weight", None)
            logging.info(f"Using cross-entropy with weights: {weight}")
            return CrossEntropyLoss(logits_ndim=3, weight=weight)
        else:
            return CrossEntropyLoss(logits_ndim=3)

    def _setup_dataloader_from_config(self, config: DictConfig):
        OmegaConf.set_struct(config, False)
        config.is_regression_task = self.is_regression_task
        OmegaConf.set_struct(config, True)
        shuffle = config.get('shuffle', False)

        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                raise ValueError(
                    "Could not load dataset as `manifest_filepath` is None or "
                    f"`tarred_audio_filepaths` is None. Provided cfg : {config}"
                )

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = get_tarred_audio_multi_label_dataset(
                cfg=config, shuffle_n=shuffle_n, global_rank=self.global_rank, world_size=self.world_size,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                raise ValueError(f"Could not load dataset as `manifest_filepath` is None. Provided cfg : {cfg}")
            dataset = get_audio_multi_label_dataset(config)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size", 1),
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def get_label_masks(self, labels, labels_len):
        mask = torch.arange(labels.size(1))[None, :].to(labels.device) < labels_len[:, None]
        return mask.to(labels.device, dtype=bool)

    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )

        # Crop or pad is always applied
        if self.crop_or_pad is not None:
            processed_signal, processed_signal_len = self.crop_or_pad(
                input_signal=processed_signal, length=processed_signal_len
            )
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_len)
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        logits = self.decoder(encoded.transpose(1, 2))
        return logits

    # PTL-specific methods
    def training_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        labels, labels_len = self.reshape_labels(logits, labels, labels_len)
        masks = self.get_label_masks(labels, labels_len)

        loss_value = self.loss(logits=logits, labels=labels, loss_mask=masks)

        tensorboard_logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        metric_logits, metric_labels = self.get_metric_logits_labels(logits, labels, masks)
        self._accuracy(logits=metric_logits, labels=metric_labels)
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_logs[f'training_batch_accuracy_top@{top_k}'] = score

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        labels, labels_len = self.reshape_labels(logits, labels, labels_len)
        masks = self.get_label_masks(labels, labels_len)

        loss_value = self.loss(logits=logits, labels=labels, loss_mask=masks)

        metric_logits, metric_labels = self.get_metric_logits_labels(logits, labels, masks)

        acc = self._accuracy(logits=metric_logits, labels=metric_labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k

        self._macro_accuracy.update(preds=metric_logits, target=metric_labels)
        stats = self._macro_accuracy._get_final_stats()

        return {
            f'{tag}_loss': loss_value,
            f'{tag}_correct_counts': correct_counts,
            f'{tag}_total_counts': total_counts,
            f'{tag}_acc_micro': acc,
            f'{tag}_acc_stats': stats,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0, tag: str = 'val'):
        val_loss_mean = torch.stack([x[f'{tag}_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x[f'{tag}_correct_counts'] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x[f'{tag}_total_counts'] for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()

        self._macro_accuracy.tp = torch.stack([x[f'{tag}_acc_stats'][0] for x in outputs]).sum(axis=0)
        self._macro_accuracy.fp = torch.stack([x[f'{tag}_acc_stats'][1] for x in outputs]).sum(axis=0)
        self._macro_accuracy.tn = torch.stack([x[f'{tag}_acc_stats'][2] for x in outputs]).sum(axis=0)
        self._macro_accuracy.fn = torch.stack([x[f'{tag}_acc_stats'][3] for x in outputs]).sum(axis=0)
        macro_accuracy_score = self._macro_accuracy.compute()

        self._accuracy.reset()
        self._macro_accuracy.reset()

        tensorboard_log = {
            f'{tag}_loss': val_loss_mean,
            f'{tag}_acc_macro': macro_accuracy_score,
        }

        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_log[f'{tag}_acc_micro_top@{top_k}'] = score

        self.log_dict(tensorboard_log, sync_dist=True)
        return tensorboard_log

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, tag='test')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_validation_epoch_end(outputs, dataloader_idx, tag='test')

    def reshape_labels(self, logits, labels, labels_len):
        logits_max_len = logits.size(1)
        labels_max_len = labels.size(1)
        batch_size = logits.size(0)
        if logits_max_len < labels_max_len:
            ratio = labels_max_len // logits_max_len
            res = labels_max_len % logits_max_len
            if res > 0:
                labels = labels[:, :-res]
                mask = labels_len > (labels_max_len - res)
                labels_len = labels_len - mask * (labels_len - (labels_max_len - res))
            labels = labels.view(batch_size, ratio, -1).amax(1)
            labels_len = torch.div(labels_len, ratio, rounding_mode="floor")
            return labels.contiguous(), labels_len.contiguous()
        elif logits_max_len > labels_max_len:
            ratio = logits_max_len / labels_max_len
            res = logits_max_len % labels_max_len
            if ceil(ratio) - ratio < 0.1:  # e.g., ratio is 1.93
                labels = labels.repeat_interleave(ceil(ratio), dim=1).long()
                labels = labels[:, :logits_max_len]
                labels_len = labels_len * ceil(ratio)
                mask = labels_len > logits_max_len
                labels_len = labels_len - mask * (labels_len - logits_max_len)
            else:  # e.g., ratio is 2.01
                labels = labels.repeat_interleave(floor(ratio), dim=1).long()
                labels_len = labels_len * floor(ratio)
                if res > 0:
                    labels = torch.cat([labels, labels[:, -res:]], dim=1)
                    labels_len = labels_len  # ignore additional "res" labels
            return labels.contiguous(), labels_len.contiguous()
        else:
            return labels, labels_len

    def get_metric_logits_labels(self, logits, labels, masks):
        """
        Params:
        -   logits: tensor of shape [B, T, C]
        -   labels: tensor of shape [B, T]
        -   masks: tensor of shape [B, T]
        Returns:
        -   logits of shape [N, C]
        -   labels of shape [N,]
        """
        C = logits.size(2)
        logits = logits.view(-1, C)  # [BxT, C]
        labels = labels.view(-1).contiguous()  # [BxT,]
        masks = masks.view(-1)  # [BxT,]
        idx = masks.nonzero()  # [BxT, 1]

        logits = logits.gather(dim=0, index=idx.repeat(1, 2))
        labels = labels.gather(dim=0, index=idx.view(-1))

        return logits, labels
