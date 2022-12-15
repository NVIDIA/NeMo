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
from nemo.collections.tts.modules.transformer import FFTransformerDecoder, BiModalTransformerEncoder



import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Linear, MSELoss
import matplotlib.pyplot as plt

from nemo.collections.tts.modules.transformer import (
    BiModalTransformerEncoder,
    FFTransformerDecoder
)
from nemo.core.classes import (
    Exportable,
    Loss,
    ModelPT,
    NeuralModule,
    typecheck
)
from nemo.core.neural_types.elements import (
    LossType,
    MelSpectrogramType,
)
from nemo.core.neural_types.neural_type import NeuralType


class BaselineModule(NeuralModule):
    def __init__(
        self,
        num_mels,
        text_encoder, spectrogram_encoder,
        infuser_model,
    ):
        """TODO"""
        super().__init__()
        self.text_encoder = text_encoder

        # todo experiment using the same params for spectrogram projection
        internal_dim = spectrogram_encoder.d_model
        self.spectrogram_in_projection = Linear(num_mels, internal_dim)
        self.spectrogram_out_projection = Linear(internal_dim, num_mels)

        self.spectrogram_encoder = spectrogram_encoder
        self.infuser_model = infuser_model

    def forward(
        self, transcripts, transcripts_len, input_mels, input_mels_len
    ):
        """TODO"""
        # TODO check masking is done correctly here
        text_encodings, _ = self.text_encoder(transcripts)

        spectrogram_projections = self.spectrogram_in_projection(input_mels)
        spectrogram_encodings, _ = self.spectrogram_encoder(
            input=spectrogram_projections, seq_lens=input_mels_len)

        text_encs, spec_encs = self.infuser_model(
            text_encs=text_encodings,
            text_lens=transcripts_len,
            spec_encs=spectrogram_encodings,
            spec_lens=input_mels_len,
        )

        spectrogram_preds = self.spectrogram_out_projection(spec_encs)
        return spectrogram_preds


class InpaintingMSELoss(Loss):
    @property
    def input_types(self):
        return {
            "spect_predicted": NeuralType(('B', 'T', 'D'), MelSpectrogramType()),
            "spect_tgt": NeuralType(('B', 'T', 'D'), MelSpectrogramType()),
            "spect_mask": NeuralType(('B', 'T', 'D'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, spect_predicted, spect_tgt, spect_mask):
        spect_tgt.requires_grad = False

        # calculate the error in the gap
        gap_mask = (1 - spect_mask)

        gap_target = spect_tgt * gap_mask
        gap_predicted = spect_predicted * gap_mask

        loss_fn = F.mse_loss
        mse_loss = loss_fn(gap_predicted, gap_target, reduction='none')

        gap_error = mse_loss.sum() / spect_predicted.shape[0]

        # TODO - could look into different ways of reducing the per-pixel
        # loss here such as averaging per example in the batch

        # TODO - loss term for
        # * whole spectrogram
        # * coefficient * (just reconstructed part)

        return gap_error


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

        text_encoder = instantiate(cfg.input_fft, **input_fft_kwargs)

        # todo separate hparams for spectrogram encoder
        spectrogram_encoder = FFTransformerDecoder(
            n_layer=cfg.input_fft.n_layer,
            n_head=cfg.input_fft.n_head,
            d_model=cfg.input_fft.d_model,
            d_head=cfg.input_fft.d_head,
            d_inner=cfg.input_fft.d_inner,
            kernel_size=cfg.input_fft.kernel_size,
            dropout=cfg.input_fft.dropout,
            dropatt=cfg.input_fft.dropatt,
            dropemb=cfg.input_fft.dropemb,
        )

        # todo hparams for infuser model
        infuser_model = BiModalTransformerEncoder(
            n_layer=cfg.input_fft.n_layer,
            n_head=cfg.input_fft.n_head,
            d_model=cfg.input_fft.d_model,
            d_head=cfg.input_fft.d_head,
            d_inner=cfg.input_fft.d_inner,
            kernel_size=cfg.input_fft.kernel_size,
            dropout=cfg.input_fft.dropout,
            dropatt=cfg.input_fft.dropatt,
            dropemb=cfg.input_fft.dropemb,
            d_mode=cfg.input_fft.d_model // 4
        )

        # output_decoder = instantiate(cfg.output_fft)
        self.module = BaselineModule(
            cfg.n_mel_channels,
            text_encoder, spectrogram_encoder,
            infuser_model,
        )

        # TODO - review this loss to only look at the reconstructed part
        self.gap_loss = InpaintingMSELoss()
        self.mse_loss = MSELoss(reduction='mean')
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

    def normalize_text(self, text):
        return self.text_normalizer_call(
            text, **self.text_normalizer_call_kwargs)

    def make_spectrogram(self, audio):
        (spectrogram,), _ = self.preprocessor(
            input_signal=torch.tensor([audio]),
            length=torch.tensor([len(audio)])
        )
        spectrogram = spectrogram.transpose(0, 1)
        return spectrogram

    def fill_in_audio_gap(
        self, *,
        audio_before=None, audio_after=None,
        full_transcript=None,
        gap_duration=None,
    ):
        """"
        Model inference

        NOTE: make sure the audio_before and auidio_after arrays are
            of the correct sample rate. Otherwise bad things happen

        Inputs:
            audio_before: array of samples [-1, 1] of audio before
            audio_after
            full_transcript
            gap_duration
        """
        assert full_transcript is not None, 'no transcript given'
        assert gap_duration is not None, 'no gap duration given'
        assert audio_before is not None or audio_after is not None, (
            'must be run with some audio')

        text_normalized = self.normalize_text(full_transcript)
        tokens = self.vocab(text_normalized)

        # Get spectrograms of left and right
        spectrogram_before = None
        if audio_before is not None:
            spectrogram_before = self.make_spectrogram(audio_before)
            num_mels = spectrogram_before.shape[1]

        spectrogram_after = None
        if audio_after is not None:
            spectrogram_after = self.make_spectrogram(audio_after)
            num_mels = spectrogram_after.shape[1]

        ffts_per_sec = self.audio_sample_rate / self.spectrogram_sample_stride
        silence_bit = torch.zeros(
            size=(int(gap_duration * ffts_per_sec), num_mels))

        # merge all the spectrograms together
        concatenate_tuple = [
            x for x in [spectrogram_before, silence_bit, spectrogram_after]
            if x is not None
        ]
        full_spectrogram = torch.concatenate(concatenate_tuple, axis=0)
        full_spectrogram = full_spectrogram.type(torch.float32)

        predicted_spectrogram, = self(
            transcripts=torch.tensor([tokens]),
            transcripts_len=torch.tensor([len(tokens)]),
            input_mels=full_spectrogram.unsqueeze(0),
            input_mels_len=torch.tensor([full_spectrogram.shape[0]])
        )
        return predicted_spectrogram

    def forward_pass(self, batch, batch_idx):
        audio, audio_lens, text, text_lens, durs, pitch, speaker = batch

        mels, spec_len = self.preprocessor(
            input_signal=audio, length=audio_lens)
        mels = mels.transpose(2, 1)  # B x T x E

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
        gap_loss = self.gap_loss(
            spect_predicted=mels_pred, spect_tgt=mels,
            spect_mask=mel_masks
        )
        mse_loss = self.mse_loss(mels_pred, mels)
        loss = gap_loss + mse_loss
        return loss, (text, text_lens, mels, mels_masked, mels_pred, spec_len)

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward_pass(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, specs = self.forward_pass(batch, batch_idx)
        (texts, text_lens, mels, mels_masked, mels_pred, mels_len) = specs
        self.log("validation_loss", loss)
        return {
            'loss': loss,
            'texts': texts,
            'text_lens': text_lens,
            'input_spectrogram': mels,
            'input_spectrogram_masked': mels_masked,
            'pred_spectrogram': mels_pred,
            'spectrogram_lens': mels_len
        }

    def validation_epoch_end(self, outputs):
        losses = [o['loss'] for o in outputs]
        self.log("validation_mean_loss", sum(losses) / len(losses))
        if self.tb_logger is None or len(outputs) == 0:
            return

        batch = outputs[0]
        for i in range(3):  # log the first 3 examples in the validation setting
            l = batch['spectrogram_lens'][i]  # noqa

            input_spectrogram = batch['input_spectrogram'][i][:l]
            input_spectrogram_masked = batch['input_spectrogram_masked'][i][:l]
            pred_spectrogram = batch['pred_spectrogram'][i][:l]

            f, axarr = plt.subplots(3)
            f.suptitle('input text TBD')
            axarr[0].imshow(input_spectrogram.cpu().numpy().T)
            axarr[1].imshow(input_spectrogram_masked.cpu().numpy().T)
            axarr[2].imshow(pred_spectrogram.cpu().numpy().T)
            self.tb_logger.add_figure(
                f'validation_{i+1}', f, global_step=self.global_step)

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

            mask[i, start:start + mask_len] = 0

        return mask

    @property
    def tb_logger(self):
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                return logger.experiment

        return None
