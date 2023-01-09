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
from nemo.collections.tts.helpers.helpers import binarize_attention_parallel, regulate_len
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss


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
from nemo.collections.tts.losses.fastpitchloss import MelLoss


class BidirectionalLinear(Linear):

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, bias=False)

    def reverse(self, input):
        return torch.matmul(input, self.weight)


class BaselineModule(NeuralModule):
    def __init__(
        self,
        num_mels,
        text_encoder,
        aligner,
        decoder,
    ):
        """TODO"""
        super().__init__()
        self.text_encoder = text_encoder

        internal_dim = decoder.d_model - text_encoder.d_model
        self.spectrogram_projection = BidirectionalLinear(
            num_mels, internal_dim)

        self.aligner = aligner

        self.decoder = decoder

    def forward(
        self, transcripts, transcripts_len, input_mels, input_mels_len,
        align_prior=None
    ):
        """TODO"""
        # TODO check masking is done correctly here
        text_encodings, text_mask = self.text_encoder(transcripts)

        spectrogram_projections = self.spectrogram_projection(input_mels)

        attn_soft, attn_logprob = self.aligner(
            queries=input_mels.permute(0, 2, 1),  # aligner assumes time last axis
            keys=text_encodings.permute(0, 2, 1),
            mask=text_mask == 0,
            attn_prior=align_prior
        )
        attn_hard = binarize_attention_parallel(
            attn_soft, transcripts_len, input_mels_len)
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]

        # repeat each text encoding the number of mel FFTs the model expects
        text_encodings_repeated, _ = regulate_len(
            attn_hard_dur, text_encodings, pace=1.0)

        both_encodings = torch.concatenate(
            (spectrogram_projections, text_encodings_repeated), axis=-1)

        output_decodings, _ = self.decoder(
            input=both_encodings,
            seq_lens=input_mels_len
        )
        spec_encoding_size = spectrogram_projections.shape[-1]
        output_decodings_trimmed = output_decodings[:, :, :spec_encoding_size]

        spectrogram_preds = self.spectrogram_projection.reverse(
            output_decodings_trimmed)

        alignment_outputs = (attn_logprob, attn_soft, attn_hard)

        return spectrogram_preds, alignment_outputs


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

        avg_pixel_loss = mse_loss.sum() / gap_mask.sum()
        return avg_pixel_loss


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
        text_encoder_kwargs = {
            "n_embed": len(self.vocab.tokens),
            "padding_idx": self.vocab.pad,
        }

        aligner = instantiate(cfg.alignment_module)
        self.forward_sum_loss_fn = ForwardSumLoss()
        self.bin_loss_fn = BinLoss()
        self.bin_loss_warmup_epochs = cfg.bin_loss_warmup_epochs

        text_encoder = instantiate(cfg.text_encoder, **text_encoder_kwargs)
        decoder = instantiate(cfg.decoder)

        self.module = BaselineModule(
            num_mels=cfg.n_mel_channels,
            text_encoder=text_encoder,
            aligner=aligner,
            decoder=decoder,
        )

        # TODO - review this loss to only look at the reconstructed part
        self.gap_loss = InpaintingMSELoss()
        self.mse_loss = MelLoss()
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

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

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

        (predicted_spectrogram,), _ = self(
            transcripts=torch.tensor([tokens]),
            transcripts_len=torch.tensor([len(tokens)]),
            input_mels=full_spectrogram.unsqueeze(0),
            input_mels_len=torch.tensor([full_spectrogram.shape[0]])
        )
        return predicted_spectrogram

    def forward_pass(self, batch, batch_idx):
        (
            audio, audio_lens,
            text, text_lens,
            align_prior_matrix,
            pitch, pitch_lens
        ) = batch

        mels, spec_len = self.preprocessor(
            input_signal=audio, length=audio_lens)
        mels = mels.transpose(2, 1)  # B x T x E

        # mask the spectrogram rather than audio to avoid artifacts
        # at the edge of the spectrogram
        mel_masks = self.make_spectrogram_mask(mels, spec_len)
        mels_masked = mels * mel_masks

        mels_pred, alignment_outputs = self(
            transcripts=text,
            transcripts_len=text_lens,
            input_mels=mels_masked,
            input_mels_len=spec_len,
            align_prior=align_prior_matrix
        )

        gap_loss = self.gap_loss(
            spect_predicted=mels_pred, spect_tgt=mels,
            spect_mask=mel_masks
        )
        mse_loss = self.mse_loss(
            spect_predicted=mels_pred, spect_tgt=mels)
        # TODO make this a parameter
        reconstruction_loss = (10 * gap_loss) + mse_loss

        (attn_logprob, attn_soft, attn_hard) = alignment_outputs
        ctc_loss = self.forward_sum_loss_fn(attn_logprob=attn_logprob, in_lens=text_lens, out_lens=spec_len)
        bin_loss_weight = min(self.current_epoch / self.bin_loss_warmup_epochs, 1.0) * 1.0
        bin_loss = self.bin_loss_fn(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
        alignment_loss = ctc_loss + bin_loss

        outputs = (text, text_lens, mels, mels_masked, mels_pred, spec_len, attn_soft)

        return reconstruction_loss, alignment_loss, outputs

    def training_step(self, batch, batch_idx):
        loss_spec, loss_alignment, specs = self.forward_pass(batch, batch_idx)
        train_loss = loss_spec + loss_alignment
        self.log("train_loss", train_loss)

        if self.log_train_spectrograms:
            (texts, text_lens, mels, mels_masked, mels_pred, mels_len, attn_soft) = specs
            self._log_spectrograms(
                mels_len[:3],
                mels[:3],
                mels_masked[:3],
                mels_pred[:3],
                attn_soft[:3],
                'train'
            )
            self.log_train_spectrograms = False

        return train_loss

    def validation_step(self, batch, batch_idx):
        loss_spec, loss_alignment, specs = self.forward_pass(batch, batch_idx)
        (texts, text_lens, mels, mels_masked, mels_pred, mels_len, attn_soft) = specs
        self.log("validation_loss", loss_spec + loss_alignment)
        return {
            'loss_spec': loss_spec,
            'loss_alignment': loss_alignment,
            'texts': texts,
            'text_lens': text_lens,
            'input_spectrogram': mels,
            'input_spectrogram_masked': mels_masked,
            'pred_spectrogram': mels_pred,
            'spectrogram_lens': mels_len,
            'attn_soft': attn_soft
        }

    def validation_epoch_end(self, outputs):
        losses = [o['loss_spec'] + o['loss_alignment'] for o in outputs]
        self.log("validation_mean_loss", sum(losses) / len(losses))
        if self.tb_logger is None or len(outputs) == 0:
            return

        batch = outputs[0]
        self._log_spectrograms(
            batch['spectrogram_lens'][:3],
            batch['input_spectrogram'][:3],
            batch['input_spectrogram_masked'][:3],
            batch['pred_spectrogram'][:3],
            batch['attn_soft'][:3],
            'validation'
        )
        self.log_train_spectrograms = True

    def _log_spectrograms(
        self,
        spectrogram_lens,
        input_spectrograms,
        input_spectrograms_masked,
        pred_spectrograms,
        attn_soft,
        name_prefix
    ):
        for i, (
            l,
            input_spectrogram,
            input_spectrogram_masked,
            pred_spectrogram,
            text_alignment,
        ) in enumerate(zip(
            spectrogram_lens,
            input_spectrograms,
            input_spectrograms_masked,
            pred_spectrograms,
            attn_soft,
        )):
            f, axarr = plt.subplots(4)
            f.suptitle('input text TBD')
            axarr[0].imshow(input_spectrogram[:l].cpu().numpy().T)
            axarr[1].imshow(input_spectrogram_masked[:l].cpu().numpy().T)
            axarr[2].imshow(text_alignment[:l].cpu().detach().numpy().T)
            axarr[3].imshow(pred_spectrogram[:l].cpu().detach().numpy().T)
            self.tb_logger.add_figure(
                f'{name_prefix}_{i+1}', f, global_step=self.global_step)

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
        max_duration_frames = int(ffts_per_sec * 1)

        for i, spec_len in enumerate(mel_lens):
            start = random.randint(
                0,
                # make sure some part of the spectrogram is masked
                spec_len - (min_duration_frames // 2)
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


class ConvDiscriminator(NeuralModule):
    def __init__(self, num_mels, window_size):

        pass


class ConvUnit(NeuralModule):
    def __init__(self, input_dims, c, s_t, s_f):
        """TODO"""
        in_channels, input_width, input_height = input_dims
        # first part:
        super().__init__()
        self.elu = torch.nn.ELU()
        self.conv_1_1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=c,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.norm_1_1 = torch.nn.LayerNorm((c, input_width, input_height))

        self.conv_1_2 = torch.nn.Conv2d(
            in_channels=c,
            out_channels=2 * c,
            kernel_size=(s_t + 2, s_f + 2),
            stride=(s_t, s_f),
            padding=(1, 1)
        )
        self.norm_1_2 = torch.nn.LayerNorm(
            (c * 2, input_width // 2, input_height // 2))

        # second part
        self.pooling = torch.nn.AvgPool2d(
            kernel_size=(s_t, s_f),
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * c,
            kernel_size=(1, 1),
            padding='same'
        )
        self.norm_2 = torch.nn.LayerNorm(
            (c * 2, input_width // 2, input_height // 2))

    def forward(self, x):
        # part 1
        h = self.norm_1_1(self.conv_1_1(self.elu(x)))
        part_1 = self.norm_1_2(self.conv_1_2(self.elu(h)))

        # part 2
        part_2 = self.norm_2(self.conv_2(self.pooling(x)))
        return part_1 + part_2
