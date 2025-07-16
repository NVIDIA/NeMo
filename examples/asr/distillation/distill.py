import logging

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import MSELoss

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models import ASRModel, EncDecCTCModel
from nemo.core.classes.mixins import AccessMixin
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


class KDAdapter(torch.nn.Module):
    """Adapter for knowledge distillation. This module adapts the teacher's features to match the student's features."""

    def __init__(self, in_channels, out_channels, kernel_size=1):
        """

        Args:
            in_channels: the number of input channels from the teacher model.
            out_channels: the number of output channels for the student model.
            kernel_size: the size of the convolution kernel to use for adaptation.
        """
        super().__init__()
        self.adapter = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.kd_loss_fn = MSELoss(reduction="mean")

    def forward(self, t_feats, s_feats):
        x = t_feats.permute(0, 2, 1)
        x = self.adapter(x)
        x = torch.nn.functional.interpolate(x, size=s_feats.size(1), mode='linear', align_corners=False)
        proj_t = x.permute(0, 2, 1)
        return self.kd_loss_fn(proj_t, s_feats)


class KnowledgeDistillationCTCModule(EncDecCTCModel):
    def __init__(self, cfg: DictConfig, trainer=None):
        super(KnowledgeDistillationCTCModule, self).__init__(cfg, trainer)

        if not hasattr(cfg, "student_model_path"):
            raise ValueError("Knowledge distillation config file must have `student_model_path`")
        if not hasattr(cfg, "teacher_model_path"):
            raise ValueError("Knowledge distillation config file must have `teacher_model_path`")

        student_model_path = cfg.get("student_model_path", None)
        if student_model_path is None:
            raise ValueError("`student_model_path` cannot be None")

        student_model = ASRModel.restore_from(student_model_path)

        # Restore checkpoint into current model
        self.load_state_dict(student_model.state_dict(), strict=False)
        logging.info(f'Student model checkpoint restored from nemo file with path : `{student_model_path}`')
        del student_model

        teacher_model_path = cfg.get("teacher_model_path", None)
        if teacher_model_path is None:
            raise ValueError("`teacher_model_path` cannot be None")

        self.teacher = ASRModel.restore_from(restore_path=teacher_model_path)
        self.teacher.eval()
        if not hasattr(cfg, "kd_pairs"):
            raise ValueError("Knowledge distillation config file must have `kd_pairs`")

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.kd_pairs = cfg.get("kd_pairs", None)
        if self.kd_pairs is None:
            raise ValueError("`kd_pairs` cannot be None")

        self.kd_loss_weight = cfg.get("kd_loss_weight", 1.0)

        # Dictionaries to store intermediate outputs from hooks
        self.teacher_outputs = {}
        self.student_outputs = {}

        # Register forward hooks for specified layers
        t_modules = dict(self.teacher.named_modules())
        s_modules = dict(self.named_modules())
        for pair in self.kd_pairs:
            t_name = pair["teacher"]
            s_name = pair["student"]
            if t_name not in t_modules:
                raise ValueError(f"Teacher layer '{t_name}' not found.")
            if s_name not in s_modules:
                raise ValueError(f"Student layer '{s_name}' not found.")
            t_modules[t_name].register_forward_hook(self._get_hook(self.teacher_outputs, t_name))
            s_modules[s_name].register_forward_hook(self._get_hook(self.student_outputs, s_name))

        self.adapters = torch.nn.ModuleDict()
        for pair in self.kd_pairs:
            self.adapters[pair["student"].replace(".", "_")] = KDAdapter(pair["t_dim"], pair["s_dim"])

    def _get_hook(self, output_dict, name):
        """Returns a hook that stores the layer's output in output_dict."""

        def hook(module, input, output):
            output_dict[name] = output

        return hook

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch
        is_dali = isinstance(batch, DALIOutputs) and batch.has_processed_signal

        # clear previous hook outputs
        self.teacher_outputs.clear()
        self.student_outputs.clear()

        # teacher forward-pass
        with torch.no_grad():
            if is_dali:
                t_log_probs, t_encoded_len, t_predictions = self.teacher.forward(
                    processed_signal=signal, processed_signal_length=signal_len
                )
            else:
                t_log_probs, t_encoded_len, t_predictions = self.teacher.forward(
                    input_signal=signal, input_signal_length=signal_len
                )

        # student forward-pass
        if is_dali:
            log_probs, encoded_len, predictions = super().forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = super().forward(input_signal=signal, input_signal_length=signal_len)

        loss_value = self.loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )

        kd_loss = 0.0
        for pair in self.kd_pairs:
            t_feat = self.teacher_outputs[pair["teacher"]]
            s_feat = self.student_outputs[pair["student"]]
            kd_loss += self.adapters[pair["student"].replace(".", "_")](t_feat, s_feat)

        kd_loss = kd_loss * self.kd_loss_weight

        loss_value = loss_value + kd_loss

        loss_value = self.add_auxiliary_losses(loss_value)
        log_every_n = getattr(self._trainer, "log_every_n_steps", 1)
        loss_value, tb_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=((batch_nb + 1) % log_every_n == 0),
        )

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # 7) logging
        tb_logs.update(
            {
                "train_loss": loss_value,
                "kd_loss": kd_loss,
                "learning_rate": self._optimizer.param_groups[0]["lr"],
                "global_step": torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )
        if (batch_nb + 1) % log_every_n == 0:
            self.wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self.wer.compute()
            self.wer.reset()
            tb_logs["training_batch_wer"] = wer

        return {"loss": loss_value, "log": tb_logs}


@hydra_runner(config_path="conf/asr_distillation", config_name="distill")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    kd_model = KnowledgeDistillationCTCModule(cfg=cfg.model, trainer=trainer)

    kd_model.setup_training_data(cfg.model.train_ds)
    kd_model.setup_multiple_validation_data(cfg.model.validation_ds)

    kd_model.setup_optimization(cfg.model.optim)

    trainer.fit(kd_model)


if __name__ == "__main__":
    main()
