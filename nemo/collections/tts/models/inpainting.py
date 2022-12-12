"""
NeMo model for audio inpainting

It uses a TTSDataset to get the preprocessed audio data and then randomly
erases a window of the spectrogram for the module to regenerate.
"""
import contextlib
from nemo.core.classes import ModelPT
from nemo.core.classes import Exportable
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from nemo.core.classes import NeuralModule, typecheck
from nemo.collections.tts.losses.fastpitchloss import MelLoss
from nemo.core.classes import Loss
import torch
import logging
import torch.nn.functional as F
import random
from nemo.core.neural_types.elements import (
    LengthsType,
    LossType,
    MelSpectrogramType,
    RegressionValuesType,
    TokenDurationType,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType


class BaselineModule(NeuralModule):
    def __init__(self):
        """TODO"""
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding='same',
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
        )

    def forward(
        self, transcripts, transcripts_len, input_mels, input_mels_len
    ):
        """TODO"""
        input_mels = input_mels.unsqueeze(1)
        return self.conv(input_mels).squeeze(1)


class InpaintingMSELoss(Loss):
    @property
    def input_types(self):
        return {
            "spect_predicted": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spect_tgt": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spect_mask": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, spect_predicted, spect_tgt, spect_mask):
        spect_tgt.requires_grad = False
        spect_tgt = spect_tgt.transpose(1, 2)  # (B, T, H)
        spect_predicted = spect_predicted.transpose(1, 2)  # (B, T, H)
        spect_mask = spect_mask.transpose(1, 2)  # (B, T, H)

        gap_mask = (1 - spect_mask)

        gap_target = spect_tgt * gap_mask
        gap_predicted = spect_predicted * gap_mask

        loss_fn = F.mse_loss
        mse_loss = loss_fn(gap_predicted, gap_target, reduction='none')

        # TODO - could look into different ways of reducing the per-pixel
        # loss here such as averaging per example in the batch

        return mse_loss.sum() / spect_predicted.shape[0]


class InpainterModel(ModelPT, Exportable):
    """TODO"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.normalizer = self._setup_normalizer(cfg)
        self.text_normalizer_call = self.normalizer.normalize
        if "text_normalizer_call_kwargs" in cfg:
            self.text_normalizer_call_kwargs = cfg.text_normalizer_call_kwargs
        self.vocab = self._setup_tokenizer(cfg)

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = instantiate(cfg.preprocessor)
        input_fft_kwargs = {
            "n_embed": len(self.vocab.tokens),
            "padding_idx": self.vocab.pad,
        }

        # text_encoder = instantiate(cfg.input_fft, **input_fft_kwargs)
        # output_fft = instantiate(cfg.output_fft)
        # self.module = BaselineModule(text_encoder, output_fft)
        self.module = BaselineModule()

        # TODO - review this loss to only look at the reconstructed part
        self.mel_loss = InpaintingMSELoss()
        self.audio_sample_rate = cfg.sample_rate
        self.spectrogram_sample_stride = cfg.n_window_stride


    def _setup_normalizer(self, cfg):
        if "text_normalizer" in cfg:
            normalizer_kwargs = {}

            if "whitelist" in cfg.text_normalizer:
                normalizer_kwargs["whitelist"] = self.register_artifact(
                    'text_normalizer.whitelist', cfg.text_normalizer.whitelist
                )

            try:
                normalizer = instantiate(cfg.text_normalizer, **normalizer_kwargs)
            except Exception as e:
                logging.error(e)
                raise ImportError(
                    "`pynini` not installed, please install via NeMo/nemo_text_processing/pynini_install.sh"
                )

            return normalizer

    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}

        if "g2p" in cfg.text_tokenizer:
            g2p_kwargs = {}

            if "phoneme_dict" in cfg.text_tokenizer.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', cfg.text_tokenizer.g2p.heteronyms,
                )

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        return instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)

    def forward(self, transcripts, transcripts_len, input_mels, input_mels_len):
        return self.module(
            transcripts, transcripts_len, input_mels, input_mels_len)

    def forward_pass(self, batch, batch_idx):
        audio, audio_lens, text, text_lens, durs, pitch, speaker = batch

        mels, spec_len = self.preprocessor(
            input_signal=audio, length=audio_lens)

        # mask the spectrogram rather than audio to avoid artifacts
        # at the edge of the spectrogram
        mel_masks = self.make_spectrogram_mask(mels, spec_len)
        mels_masked = mels * mel_masks

        mels_pred = self(
            transcripts=text,
            transcripts_len=text_lens,
            input_mels=mels_masked,
            input_mels_len=spec_len
        )
        loss = self.mel_loss(spect_predicted=mels_pred, spect_tgt=mels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward_pass(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step_step(self, batch, batch_idx):
        loss = self.forward_pass(batch, batch_idx)
        self.log("validation_loss", loss)
        return loss

    def validation_epoch_end(self, outputs):
        self.log("validation_mean_loss", outputs.mean())

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(
            cfg, shuffle_should_be=False, name="val")

    def __setup_dataloader_from_config(
        self, cfg, shuffle_should_be: bool = True, name: str = "train"
    ):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader "
                    "but was not found in its  config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(
                    f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif cfg.dataloader_params.shuffle:
            logging.error(
                f"The {name} dataloader for {self} has shuffle set to True!!!")

        assert cfg.dataset._target_ == "nemo.collections.tts.torch.data.TTSDataset"
        phon_mode = contextlib.nullcontext()
        if hasattr(self.vocab, "set_phone_prob"):
            phon_mode = self.vocab.set_phone_prob(prob=None if name == "val" else self.vocab.phoneme_probability)

        with phon_mode:
            dataset = instantiate(
                cfg.dataset,
                text_normalizer=self.normalizer,
                text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
                text_tokenizer=self.vocab,
            )

        return torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        return []

    def make_spectrogram_mask(self, mels, mel_lens):
        mask = torch.ones_like(mels)  # doing in-place mutation here

        ffts_per_sec = self.audio_sample_rate / self.spectrogram_sample_stride

        # TODO - have these as parameters in the future
        min_duration_frames = int(ffts_per_sec * 0.5)
        max_duration_frames = int(ffts_per_sec * 2)

        for i, spec_len in enumerate(mel_lens):
            start = random.randint(
                0,
                # make sure some part of the spectrogram is masked
                spec_len - (max_duration_frames / 2)
            )
            mask_len = random.randint(
                min_duration_frames, max_duration_frames)

            mask[i, :, start:start + mask_len] = 0

        return mask
