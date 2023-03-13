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
import os
from nemo.core.classes import Loss
import torch
import logging
import torch.nn.functional as F
import random
from nemo.collections.tts.models.aligner import AlignerModel

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
from nemo.collections.tts.helpers.helpers import binarize_attention_parallel, regulate_len, quantize_durations
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from mel_cepstral_distance import get_metrics_mels
from nemo.core.optim import prepare_lr_scheduler


import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Linear, MSELoss
import matplotlib.pyplot as plt
from nemo.utils.app_state import AppState


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
    LogitsType
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.collections.tts.losses.fastpitchloss import MelLoss, DurationLoss, PitchLoss, EnergyLoss


class BidirectionalLinear(Linear):

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, bias=False)

    def reverse(self, input):
        return torch.matmul(input, self.weight)


def average_features(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = torch.nn.functional.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = torch.nn.functional.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg


class BaselineModule(NeuralModule):
    def __init__(
        self,
        num_mels,
        token_encoder,
        spectrogram_encoder,
        duration_predictor,
        pitch_predictor,
        pitch_emb,
        energy_predictor,
        energy_emb,
        decoder,
    ):
        """Module used for inpainting"""
        super().__init__()
        self.token_encoder = token_encoder

        self.spectrogram_encoder = spectrogram_encoder
        self.duration_predictor = duration_predictor
        self.pitch_predictor = pitch_predictor
        self.pitch_emb = pitch_emb
        self.decoder = decoder
        self.energy_predictor = energy_predictor
        self.energy_emb = energy_emb

        self.spectrogram_projection = BidirectionalLinear(
            num_mels, self.spectrogram_encoder.d_model)

        assert (
            self.spectrogram_encoder.d_model == self.token_encoder.d_model)
        assert (
            self.spectrogram_encoder.d_model + self.token_encoder.d_model ==
            self.decoder.d_model
        )

        def variable(*shape):
            t = torch.empty(shape)
            torch.nn.init.xavier_uniform_(t)
            return torch.nn.Parameter(torch.autograd.Variable(t))

        self.learned_query = variable(1, self.spectrogram_encoder.d_model)
        self.speaker_encoder = torch.nn.MultiheadAttention(
            embed_dim=self.spectrogram_encoder.d_model,
            num_heads=self.spectrogram_encoder.n_head,
            batch_first=True,
        )

        self.decoder = decoder

    def forward(
        self,
        input_mels, input_mels_len, tokens,
        token_durations,
        pitch=None,
        energy=None,
        slice_indices=None
    ):
        """Forward pass of the model

        Args:
            input_mels: mel spectrograms with blank sections to inpaint
            input_mels_len: lengths of the mels
            tokens: tokens for the whole transcription of the audio
            token_durations: the number of mel buckets each token takes
            pitch: the fundamental pitch of each token
            slice_indices: optional argument which outlines where to truncate
                input_mels for each example in the batch (ensures no accidental
                memory overflow)

        """
        spectrogram_projections = self.spectrogram_projection(input_mels)
        spectrogram_encodings, mask = self.spectrogram_encoder(
            input=spectrogram_projections, seq_lens=input_mels_len)

        token_encodings, tokens_mask = self.token_encoder(tokens)

        # generate the speaker embedding
        batch_dim = spectrogram_encodings.shape[0]

        speaker_encoding, _ = self.speaker_encoder(
            query=self.learned_query.repeat(batch_dim, 1).unsqueeze(1),
            key=spectrogram_encodings,
            value=spectrogram_encodings
        )

        # add speaker encoding to text encodings
        speakerized_token_encodings = token_encodings + speaker_encoding

        # predict_pitch
        pitch_predicted = self.pitch_predictor(
            speakerized_token_encodings, tokens_mask)

        # pitch is only given in training
        if pitch is not None:
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
        else:
            pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))

        # output of pitch predictor is B X E X L
        pitch_emb = pitch_emb.transpose(1, 2)

        speaker_pitch_token_encodings = speakerized_token_encodings + pitch_emb

        # predict energy
        energy_predicted = self.energy_predictor(
            speakerized_token_encodings, tokens_mask
        )

        if energy is not None:
            energy_emb = self.energy_emb(energy.unsqueeze(1))
        else:
            energy_emb = self.energy_emb(energy_predicted.unsqueeze(1))

        energy_emb = energy_emb.transpose(1, 2)
        conditioned_token_encodings = speaker_pitch_token_encodings + energy_emb

        # predict duration
        log_durs_pred = self.duration_predictor(
            conditioned_token_encodings, tokens_mask)

        aligned_token_encodings, _ = regulate_len(
            token_durations, conditioned_token_encodings, pace=1.0)

        # During training we slice the spectrograms to remove chance of
        # memory overflow
        if slice_indices is not None:
            aligned_token_encodings = gather_slices(
                aligned_token_encodings, slice_indices)

        assert input_mels.shape[1] == aligned_token_encodings.shape[1]

        both_encodings = torch.concatenate(
            (spectrogram_encodings, aligned_token_encodings), axis=-1)

        output_decodings, _ = self.decoder(
            input=both_encodings,
            seq_lens=input_mels_len
        )
        spec_encoding_size = spectrogram_encodings.shape[-1]
        output_decodings_trimmed = output_decodings[:, :, :spec_encoding_size]

        spectrogram_preds = self.spectrogram_projection.reverse(
            output_decodings_trimmed)

        return spectrogram_preds, pitch_predicted, energy_predicted, log_durs_pred

    def predict_token_durations(
        self, tokens, tokens_len, input_mels, input_mels_len
    ):
        spectrogram_projections = self.spectrogram_projection(input_mels)
        spectrogram_encodings, mask = self.spectrogram_encoder(
            input=spectrogram_projections, seq_lens=input_mels_len)

        token_encodings, tokens_mask = self.token_encoder(tokens)

        # generate the speaker embedding
        batch_dim = spectrogram_encodings.shape[0]

        speaker_encoding, _ = self.speaker_encoder(
            query=self.learned_query.repeat(batch_dim, 1).unsqueeze(1),
            key=spectrogram_encodings,
            value=spectrogram_encodings
        )

        # add speaker encoding to text encodings
        speakerized_token_encodings = token_encodings + speaker_encoding

        # predict_pitch
        pitch_predicted = self.pitch_predictor(
            speakerized_token_encodings, tokens_mask)

        pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))

        # output of pitch predictor is B X E X L
        pitch_emb = pitch_emb.transpose(1, 2)

        speaker_pitch_token_encodings = speakerized_token_encodings + pitch_emb

        # predict duration
        log_durs_predicted = self.duration_predictor(
            speaker_pitch_token_encodings, tokens_mask)

        durs_predicted = quantize_durations(
            torch.clamp(torch.exp(log_durs_predicted) - 1, 0, 100)).squeeze(0)

        return durs_predicted


class InpaintingMSELoss(Loss):
    def __init__(self, loss_fn=F.mse_loss):
        self.loss_fn = loss_fn
        super().__init__()

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
            "loss": NeuralType(elements_type=LogitsType()),
        }

    @typecheck()
    def forward(self, spect_predicted, spect_tgt, spect_mask):
        spect_tgt.requires_grad = False

        # calculate the error in the gap
        gap_mask = (1 - spect_mask)

        gap_target = spect_tgt * gap_mask
        gap_predicted = spect_predicted * gap_mask

        if self.loss_fn == "l1_and_l2":
            l1_loss = F.l1_loss(gap_predicted, gap_target, reduction='none')
            l2_loss = F.mse_loss(gap_predicted, gap_target, reduction='none')
            mse_loss = l1_loss + (0.5 * l2_loss)
        else:
            mse_loss = self.loss_fn(gap_predicted, gap_target, reduction='none')

        avg_pixel_loss = mse_loss.sum() / gap_mask.sum()
        return avg_pixel_loss


class DiscriminatorHingeLoss(Loss):
    @property
    def input_types(self):
        return {
            "logits_gen": NeuralType(('B'), MelSpectrogramType()),
            "logits_real": NeuralType(('B'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    def forward(self, logits_gen, logits_real):
        zero = torch.zeros(
            len(logits_gen),
            dtype=logits_real.dtype, device=logits_real.device
        )
        loss_real = torch.mean(torch.maximum(zero, 1 - logits_real))
        loss_gen = torch.mean(torch.maximum(zero, 1 + logits_gen))
        return loss_gen, loss_real


class FeatureMatchingLoss(Loss):

    def forward(self, activations_real, activations_gen):
        loss_per_layer = []
        for activation_real_layer, activation_gen_layer in zip(
            activations_real, activations_gen
        ):
            loss_per_layer += [F.l1_loss(
                activation_real_layer,
                activation_gen_layer,
                reduction='mean'
            )]

        return sum(loss_per_layer) / len(loss_per_layer)


class InpainterModel(ModelPT, Exportable):
    """Model for inpainting dictation audio"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        aligner = AlignerModel.from_pretrained("tts_en_radtts_aligner")
        self.inference = os.getenv('DATA_CAP') == '0'

        self.normalizer = aligner.normalizer

        self.text_normalizer_call = self.normalizer.normalize
        if "text_normalizer_call_kwargs" in cfg:
            self.text_normalizer_call_kwargs = cfg.text_normalizer_call_kwargs

        self.vocab = aligner.tokenizer
        super().__init__(cfg=cfg, trainer=trainer)

        self.preprocessor = aligner.preprocessor

        self.forward_sum_loss_fn = ForwardSumLoss()
        self.bin_loss_fn = BinLoss()
        self.bin_loss_warmup_epochs = cfg.bin_loss_warmup_epochs

        decoder = instantiate(cfg.decoder)
        token_encoder = instantiate(
            cfg.token_encoder, n_embed=len(self.vocab.tokens))
        spectrogram_encoder = instantiate(cfg.spectrogram_encoder)
        duration_predictor = instantiate(cfg.duration_predictor)
        pitch_predictor = instantiate(cfg.pitch_predictor)
        pitch_emb = torch.nn.Conv1d(
            1,
            cfg.symbols_embedding_dim,
            kernel_size=cfg.pitch_predictor.kernel_size,
            padding=int((cfg.pitch_predictor.kernel_size - 1) / 2),
        )
        energy_predictor = instantiate(cfg.energy_predictor)
        energy_emb = torch.nn.Conv1d(
            1,
            cfg.symbols_embedding_dim,
            kernel_size=cfg.energy_predictor.kernel_size,
            padding=int((cfg.energy_predictor.kernel_size - 1) / 2),
        )

        self.module = BaselineModule(
            num_mels=cfg.n_mel_channels,
            token_encoder=token_encoder,
            spectrogram_encoder=spectrogram_encoder,
            duration_predictor=duration_predictor,
            pitch_predictor=pitch_predictor,
            pitch_emb=pitch_emb,
            energy_predictor=energy_predictor,
            energy_emb=energy_emb,
            decoder=decoder,
        )
        self.postnet = instantiate(cfg.postnet) if 'postnet' in cfg else None

        loss_lookup = {
            'l1': F.l1_loss,
            'mse': F.mse_loss,
            'l1_and_l2': 'l1_and_l2',
        }
        reconstruction_loss = loss_lookup[cfg.loss]
        self.gap_loss_scale = cfg.gap_loss_scale

        self.min_mask_duration = cfg.min_mask_duration
        self.max_mask_duration = cfg.max_mask_duration
        self.max_mel_len_for_train = 1500

        # According to config, both are using L1 loss
        self.gap_loss = InpaintingMSELoss(reconstruction_loss)
        self.mse_loss = MelLoss(reconstruction_loss)
        self.duration_loss_fn = DurationLoss()
        self.pitch_loss_fn = PitchLoss()
        self.energy_loss_fn = EnergyLoss()

        self.aligner = aligner
        spectrogrammer = self.preprocessor.featurizer

        self.audio_sample_rate = spectrogrammer.sample_rate
        self.spectrogram_sample_stride = spectrogrammer.hop_length

        self.discriminator = None
        if 'discriminator' in cfg:
            self.discriminator = ConvDiscriminator(spectrogrammer.nfilt)
            self.discriminator_loss = DiscriminatorHingeLoss()
            self.feature_matching_loss = FeatureMatchingLoss()

            self.discriminator_warmup_steps = cfg.discriminator.warmup_steps
            self.discriminator_rampup_steps = cfg.discriminator.rampup_steps
            self.discriminator_batch_size = cfg.discriminator.batch_size

    def configure_optimizers(self):
        if self.discriminator is None:
            return super().configure_optimizers()

        optim_config = self.cfg.optim.copy()

        # Adding some fields needed to init a scheduler properly
        OmegaConf.set_struct(optim_config, False)
        scheduler_config = optim_config.pop("sched", None)
        scheduler_config['t_max_epochs'] = self.trainer.max_epochs
        scheduler_config['t_accumulate_grad_batches'] = (
            self.trainer.accumulate_grad_batches)
        scheduler_config['t_limit_train_batches'] = (
            self.trainer.limit_train_batches)

        # copied from ModelPT
        app_state = AppState()
        if app_state.data_parallel_size is not None:
            scheduler_config['t_num_workers'] = app_state.data_parallel_size
        elif app_state.model_parallel_size is None:
            scheduler_config['t_num_workers'] = (
                self.trainer.num_devices * self.trainer.num_nodes)
        else:
            scheduler_config['t_num_workers'] = (
                self.trainer.num_devices * self.trainer.num_nodes
            ) / app_state.model_parallel_size

        OmegaConf.set_struct(optim_config, True)

        optim_g = instantiate(optim_config, params=self.module.parameters(),)
        optim_d = instantiate(
            optim_config, params=self.discriminator.parameters())

        sched_g = prepare_lr_scheduler(
            optimizer=optim_g,
            scheduler_config=scheduler_config,
            train_dataloader=self._train_dl
        )
        # sched_d = prepare_lr_scheduler(
        #     optimizer=optim_d,
        #     scheduler_config=scheduler_config,
        #     train_dataloader=self._train_dl
        # )
        # basically a scheduler that does nothing
        sched_d = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optim_d,
            lr_lambda=lambda x: 1
        )

        return [optim_g, optim_d], [sched_g, sched_d]

    def forward(self, *args, **kwargs):
        spectrogram_preds, pitch_predicted, energy_predicted, log_durs_pred = (
            self.module(*args, **kwargs))  # calls L 141

        if self.postnet is not None:
            spectrogram_preds = self.postnet(
                mel_spec=spectrogram_preds.transpose(1, 2))
            spectrogram_preds = spectrogram_preds.transpose(1, 2)

        return spectrogram_preds, pitch_predicted, energy_predicted, log_durs_pred

    def normalize_text(self, text):
        return self.text_normalizer_call(
            text, **self.text_normalizer_call_kwargs)

    def make_spectrogram(self, audio):
        (spectrogram,), _ = self.preprocessor(
            input_signal=torch.tensor([audio]),
            length=torch.tensor([len(audio)])
        )
        spectrogram = spectrogram.transpose(0, 1)
        return spectrogram.float()

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

        (predicted_spectrogram,), *_ = self(
            transcripts=torch.tensor([tokens]),
            transcripts_len=torch.tensor([len(tokens)]),
            input_mels=full_spectrogram.unsqueeze(0),
            input_mels_len=torch.tensor([full_spectrogram.shape[0]])
        )
        return predicted_spectrogram

    def edit_recording(
        self,
        spectrogram,
        full_transcript,
        new_transcript
    ):
        """Generate a new spectrogram given an altered transcript

        Args:
            spectrogram: Original spectrogram of the audio
            full_transcript: transcript of original audio
            new_transcript: transcript for overwriting the audio

        Returns: full_replacement, partial_replacement where:
            full_replacement:  the spectrogram entirely generated by the model
            partial_replacement: a spectrogram where the new generated
                audio is inserted
        """
        text_normalized = self.normalize_text(full_transcript)
        tokens = torch.tensor(self.vocab(text_normalized))

        new_text_normalized = self.normalize_text(new_transcript)
        new_tokens = torch.tensor(self.vocab(new_text_normalized))

        spec_len = torch.tensor([len(spectrogram)])
        tokens_len = torch.tensor([len(tokens)])
        new_tokens_len = torch.tensor([len(new_tokens)])

        attn_soft, attn_logprob = self.aligner(
            spec=spectrogram.unsqueeze(0).transpose(1, 2),
            spec_len=spec_len,
            text=tokens.unsqueeze(0),
            text_len=tokens_len
        )

        attn_hard = binarize_attention_parallel(
            attn_soft, tokens_len, spec_len)
        attn_hard_dur = attn_hard.sum(2)[0, 0, :]

        # Find the first and last divergence in old and new transcripts
        first_divergence = None
        for i, (old_token, new_token) in enumerate(zip(tokens, new_tokens)):
            if old_token != new_token:
                first_divergence = i
                break

        assert first_divergence is not None, 'no divergence found between the two transcripts'
        # and the last
        last_divergence = None
        for i, (old_token, new_token) in enumerate(zip(reversed(tokens), reversed(new_tokens))):
            if old_token != new_token:
                last_divergence = i
                break

        assert last_divergence is not None, 'could not find a final divergence'

        # Get the predicted duration of the phonemes in the new text
        durs_predicted = self.module.predict_token_durations(
            tokens=new_tokens.unsqueeze(0),
            tokens_len=new_tokens_len,
            input_mels=spectrogram.unsqueeze(0),
            input_mels_len=spec_len
        )

        # Get the left spectrograms and phonemes
        token_durs_left = attn_hard_dur[:first_divergence]
        num_frames_left = sum(token_durs_left).int()
        tokens_left = tokens[:first_divergence]
        spec_left = spectrogram[:num_frames_left]

        # and the right
        token_durs_right = attn_hard_dur[-last_divergence:]
        num_frames_right = sum(token_durs_right).int()
        tokens_right = tokens[-last_divergence:]
        spec_right = spectrogram[-num_frames_right:]

        # create the middle empty audio and get the token durations
        tokens_middle = new_tokens[first_divergence:-last_divergence]
        token_durs_middle = durs_predicted[first_divergence:-last_divergence]
        num_frames_middle = sum(token_durs_middle)
        empty_spec_middle = torch.zeros(num_frames_middle, spectrogram.shape[1])

        blanked_spectrogram = torch.concatenate((spec_left, empty_spec_middle, spec_right))
        token_durations = torch.concatenate((token_durs_left, token_durs_middle, token_durs_right))

        # sanity check
        concat_tokens = torch.concatenate((tokens_left, tokens_middle, tokens_right))
        assert torch.equal(concat_tokens, new_tokens)

        mels_pred, *_ = self(
            input_mels=blanked_spectrogram.unsqueeze(0),
            input_mels_len=torch.tensor([len(blanked_spectrogram)]),
            tokens=new_tokens.unsqueeze(0),
            token_durations=token_durations.unsqueeze(0)
        )
        full_replacement = mels_pred[0]
        # also return a spectrogram with only the changed part from the model
        inpainted_section = full_replacement[num_frames_left:-num_frames_right]
        partial_replacement = torch.concatenate((spec_left, inpainted_section, spec_right))

        return full_replacement, partial_replacement

    def regenerate_audio(self, spectrogram, replacement_phrase):
        """Re-fill in a given part of some input audio.

        Args:
            spectrogram: mel spectrogram of the original audio
            replacement_phrase: original transcript with one square bracketed
                span showing which area to blank out e.g. "So [monday] at 8pm
                for 5 people?"


        """
        # TODO more input sanitization
        start_of_middle_span = replacement_phrase.find('[')
        end_of_middle_span = replacement_phrase.find(']')

        assert start_of_middle_span >= 0
        assert end_of_middle_span > start_of_middle_span

        left_phrase = replacement_phrase[:start_of_middle_span]
        middle_phrase = replacement_phrase[
            start_of_middle_span+1:end_of_middle_span]
        right_phrase = replacement_phrase[end_of_middle_span + 1:]

        def tokenize(text):
            text_normalized = self.normalize_text(text)
            return torch.tensor(self.vocab(text_normalized))

        # tokenizer puts 0 tokens on left and right of each phrase
        # we need to trim this
        left_tokens = tokenize(left_phrase)[:-1]
        middle_tokens = tokenize(middle_phrase)[1:-1]
        right_tokens = tokenize(right_phrase)[1:]

        tokens = torch.concatenate((left_tokens, middle_tokens, right_tokens))

        tokens_len = torch.tensor([len(tokens)])
        spec_len = torch.tensor([len(spectrogram)])

        attn_soft, attn_logprob = self.aligner(
            spec=spectrogram.unsqueeze(0).transpose(1, 2),
            spec_len=spec_len,
            text=tokens.unsqueeze(0),
            text_len=tokens_len
        )
        attn_hard = binarize_attention_parallel(
            attn_soft, tokens_len, spec_len)

        token_durations = attn_hard.sum(2)[0, 0, :]

        token_end_indices = token_durations.cumsum(0).int()

        # Find out which columns of the spectrogram we should blank
        if len(left_tokens) > 0:
            blank_start = token_end_indices[len(left_tokens) - 1]
        else:
            blank_start = 0

        blank_end = token_end_indices[len(left_tokens) + len(middle_tokens) - 1]

        blanked_spectrogram = torch.clone(spectrogram)
        blanked_spectrogram[blank_start:blank_end, :] = 0.

        mels_pred, *_ = self(
            input_mels=blanked_spectrogram.unsqueeze(0),
            input_mels_len=torch.tensor([len(blanked_spectrogram)]),
            tokens=tokens.unsqueeze(0),
            token_durations=token_durations.unsqueeze(0)
        )
        full_replacement = mels_pred[0]
        # also return a spectrogram with only the changed part from the model
        inpainted_section = full_replacement[blank_start:blank_end]
        spec_left = spectrogram[:blank_start]
        spec_right = spectrogram[blank_end:]
        partial_replacement = torch.concatenate((
            spec_left, inpainted_section, spec_right))

        mcd_full, _, _ = get_metrics_mels(
            spectrogram.cpu().detach().numpy(),
            full_replacement.cpu().detach().numpy(),
            take_log=False
        )
        mcd_partial, _, _ = get_metrics_mels(
            spectrogram.cpu().detach().numpy(),
            partial_replacement.cpu().detach().numpy(),
            take_log=False
        )

        import matplotlib.pyplot as plt
        f, axarr = plt.subplots(5)
        f.suptitle(replacement_phrase)
        axarr[0].imshow(spectrogram.cpu().numpy().T)
        axarr[1].imshow(blanked_spectrogram.cpu().numpy().T)
        axarr[2].imshow(attn_hard[0].cpu().detach().numpy().T)
        axarr[3].imshow(full_replacement.cpu().detach().numpy().T)
        axarr[4].imshow(partial_replacement.cpu().detach().numpy().T)
        plt.show()

        return full_replacement, partial_replacement, mcd_full, mcd_partial

    def forward_pass(self, batch, batch_idx, training=True):
        (
            audio, audio_lens,
            text, text_lens,
            align_prior_matrix,
            pitch, pitch_lens,
            energy, energy_lens,
        ) = batch

        mels, spec_len = self.preprocessor(
            input_signal=audio, length=audio_lens)
        mels = mels.transpose(2, 1)  # B x T x E

        # run the pretrained aligner
        attn_soft, attn_logprob = self.aligner(
            spec=mels.transpose(2, 1),  # aligner wants B X freq X time
            spec_len=spec_len,
            text=text,
            text_len=text_lens
        )

        attn_hard = binarize_attention_parallel(
            attn_soft, text_lens, spec_len)
        token_durations = attn_hard.sum(2)[:, 0, :]

        if pitch is not None:
            # Pitch during training is per spectrogram frame,
            # but during inference, it should be per character
            pitch = average_features(pitch.unsqueeze(1), token_durations).squeeze(1)

        if energy is not None:
            # energy during training is per spectrogram frame,
            energy = average_features(energy.unsqueeze(1), token_durations).squeeze(1)
            energy = torch.log(1.0 + energy)

        # slice the mels to make sure they dont flow over a given size
        mels, spec_len, slice_indices = random_slice_if_longer_than(
            mels, spec_len, self.max_mel_len_for_train)

        # mask the spectrogram rather than audio to avoid artifacts
        # at the edge of the spectrogram
        mel_masks = self.make_spectrogram_mask(mels, spec_len)
        mels_masked = mels * mel_masks

        # below code to make the blank spectrogram "quiet" instead of 0
        # intensity which is essentially max volume for a mel spectrogram
        # negative_mask = -14 * (1 - mel_masks)
        # mels_masked += negative_mask

        (
            spectrogram_preds,
            pitch_predicted,
            energy_pred,
            log_durs_pred,
        ) = self(
            input_mels=mels_masked,
            input_mels_len=spec_len,
            tokens=text,
            token_durations=token_durations,
            pitch=pitch if training else None,
            energy=energy if training else None,
            slice_indices=slice_indices
        )

        gap_loss = self.gap_loss(
            spect_predicted=spectrogram_preds, spect_tgt=mels,
            spect_mask=mel_masks
        )
        mse_loss = self.mse_loss(
            spect_predicted=spectrogram_preds, spect_tgt=mels)

        reconstruction_loss = (self.gap_loss_scale * gap_loss) + mse_loss

        dur_loss = self.duration_loss_fn(
            log_durs_predicted=log_durs_pred,
            durs_tgt=token_durations,
            len=text_lens
        )
        pitch_loss = self.pitch_loss_fn(
            pitch_predicted=pitch_predicted, pitch_tgt=pitch, len=text_lens)
        energy_loss = self.energy_loss_fn(
            energy_predicted=energy_pred, energy_tgt=energy, length=text_lens)

        outputs = (
            text, text_lens,
            mels, mels_masked,
            spectrogram_preds, spec_len,
            attn_hard
        )

        losses = (
            reconstruction_loss,
            dur_loss,
            pitch_loss,
            energy_loss
        )

        return losses, outputs

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if self.discriminator is None:
            return self.training_step_simple(batch, batch_idx)

        return self.training_step_with_discriminator(
            batch, batch_idx, optimizer_idx)

    def training_step_simple(self, batch, batch_idx):
        losses, specs = self.forward_pass(batch, batch_idx, training=True)

        reconstruction_loss, dur_loss, pitch_loss, energy_loss = losses

        train_loss = reconstruction_loss + dur_loss + pitch_loss + energy_loss
        self.log("train_loss", train_loss)
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("dur_loss", dur_loss)
        self.log("pitch_loss", pitch_loss)
        self.log("energy_loss", energy_loss)

        if self.log_train_spectrograms:
            (
                texts, text_lens,
                mels, mels_masked, mels_pred, mels_len,
                attn_hard
            ) = specs
            mcds = []
            for mel, mel_pred in zip(mels, mels_pred):
                mel_cepstral_distance, _, _ = get_metrics_mels(
                    mel.cpu().detach().numpy(),
                    mel_pred.cpu().detach().numpy(),
                    take_log=False
                )
                mcds += [mel_cepstral_distance]

            self.log('train_mcd', sum(mcds) / len(mcds))

            self._log_spectrograms(
                mels_len[:3],
                text_lens[:3],
                mels[:3],
                mels_masked[:3],
                mels_pred[:3],
                attn_hard[:3],
                'train'
            )
            self.log_train_spectrograms = False

        return train_loss

    def training_step_with_discriminator(
        self, batch, batch_idx, optimizer_idx
    ):
        losses, specs = self.forward_pass(batch, batch_idx, training=True)

        reconstruction_loss, dur_loss, pitch_loss, energy_loss = losses
        (
            text, text_lens,
            mels, mels_masked,
            spectrograms_pred, specs_len,
            attn_hard
        ) = specs

        # chop the spectrograms and predicted spectrograms into chunks of the
        # right size for the discriminator
        spec_windows_real = []
        spec_windows_gen = []
        for spec, spec_pred, spec_len in zip(
            mels, spectrograms_pred, specs_len
        ):
            window_size = self.discriminator.input_width
            num_windows = spec_len // window_size
            spec_windows_real += spec.split(window_size)[:num_windows]
            spec_windows_gen += spec_pred.split(window_size)[:num_windows]

        spec_windows_real = torch.stack(spec_windows_real)
        spec_windows_gen = torch.stack(spec_windows_gen)

        batch_size = min(len(spec_windows_gen), self.discriminator_batch_size)
        shuffle_indices = torch.randperm(
            batch_size, device=spectrograms_pred.device)

        batch_real = spec_windows_real[shuffle_indices]
        batch_gen = spec_windows_gen[shuffle_indices]

        training_inpainter = optimizer_idx % 2 == 0
        # Train Discriminator
        if not training_inpainter:
            if self.global_step < (self.discriminator_warmup_steps - 1000):
                return
            logits_real, _ = self.discriminator(batch_real.detach())
            logits_gen, _ = self.discriminator(batch_gen.detach())

            loss_gen, loss_real = self.discriminator_loss(
                logits_gen=logits_gen, logits_real=logits_real)
            loss = loss_gen + loss_real
            self.log("discriminator_loss", loss)
            self.log("disc_loss_real", loss_real)
            self.log("disc_loss_gen", loss_gen)

        # Train Inpainter
        if training_inpainter:
            supervised_losses = reconstruction_loss + dur_loss + pitch_loss + energy_loss

            _, activations_real = self.discriminator(batch_real)
            _, activations_gen = self.discriminator(batch_gen)

            feature_matching_loss = self.feature_matching_loss(
                activations_real=activations_real,
                activations_gen=activations_gen,
            )

            adversarial_amount = linear_rampup_with_warmup(
                self.global_step,
                warmup_steps=self.discriminator_warmup_steps,
                rampup_steps=self.discriminator_rampup_steps
            )
            feature_matching_loss_scaled = feature_matching_loss * adversarial_amount
            loss_inpainter = supervised_losses + feature_matching_loss_scaled
            loss = loss_inpainter

            self.log("inpainter_loss", loss)
            self.log("reconstruction_loss", reconstruction_loss)
            self.log("dur_loss", dur_loss)
            self.log("pitch_loss", pitch_loss)
            self.log("energy_loss", energy_loss)
            self.log("adversarial_amount", adversarial_amount)
            self.log("feature_matching_loss_scaled", feature_matching_loss_scaled)
            self.log("feature_matching_loss", feature_matching_loss)

        if self.log_train_spectrograms:
            mcds = []
            for mel, mel_pred in zip(mels, spectrograms_pred):
                mel_cepstral_distance, _, _ = get_metrics_mels(
                    mel.cpu().detach().numpy(),
                    mel_pred.cpu().detach().numpy(),
                    take_log=False
                )
                mcds += [mel_cepstral_distance]

            self.log('train_mcd', sum(mcds) / len(mcds))

            self._log_spectrograms(
                specs_len[:3],
                text_lens[:3],
                mels[:3],
                mels_masked[:3],
                spectrograms_pred[:3],
                attn_hard[:3],
                'train'
            )
            self.log_train_spectrograms = False

        return loss

    def validation_step(self, batch, batch_idx):
        losses, specs = self.forward_pass(batch, batch_idx, training=False)
        (
            texts, text_lens,
            mels, mels_masked, mels_pred, mels_len,
            attn_hard
        ) = specs
        reconstruction_loss, dur_loss, pitch_loss, energy_loss = losses

        validation_loss = reconstruction_loss + dur_loss + pitch_loss + energy_loss
        self.log("validation_loss", validation_loss)
        self.log("reconstruction_loss_val", reconstruction_loss)
        self.log("dur_loss_val", dur_loss)
        self.log("pitch_loss_val", pitch_loss)
        self.log("energy_loss_val", energy_loss)

        mcds = []
        for mel, mel_pred in zip(mels, mels_pred):
            mel_cepstral_distance, _, _ = get_metrics_mels(
                mel.cpu().detach().numpy(),
                mel_pred.cpu().detach().numpy(),
                take_log=False
            )
            mcds += [mel_cepstral_distance]

        mean_mcd = sum(mcds) / len(mcds)

        return {
            'loss_spec': validation_loss,
            'texts': texts,
            'text_lens': text_lens,
            'input_spectrogram': mels,
            'input_spectrogram_masked': mels_masked,
            'pred_spectrogram': mels_pred,
            'spectrogram_lens': mels_len,
            'attn_hard': attn_hard,
            'mean_mcd': mean_mcd
        }

    def validation_epoch_end(self, outputs):
        losses = [o['loss_spec'] for o in outputs]
        self.log("validation_mean_loss", sum(losses) / len(losses))

        mcds = [o['mean_mcd'] for o in outputs]
        self.log("validation_mcd", sum(mcds) / len(mcds))

        if self.tb_logger is None or len(outputs) == 0:
            return

        batch = outputs[0]
        self._log_spectrograms(
            batch['spectrogram_lens'][:3],
            batch['text_lens'][:3],
            batch['input_spectrogram'][:3],
            batch['input_spectrogram_masked'][:3],
            batch['pred_spectrogram'][:3],
            batch['attn_hard'][:3],
            'validation'
        )
        self.log_train_spectrograms = True

    def _log_spectrograms(
        self,
        spectrogram_lens,
        text_lens,
        input_spectrograms,
        input_spectrograms_masked,
        pred_spectrograms,
        attn_hard,
        name_prefix
    ):
        for i, (
            l,
            text_len,
            input_spectrogram,
            input_spectrogram_masked,
            pred_spectrogram,
            text_alignment,
        ) in enumerate(zip(
            spectrogram_lens,
            text_lens,
            input_spectrograms,
            input_spectrograms_masked,
            pred_spectrograms,
            attn_hard,
        )):
            f, axarr = plt.subplots(4)
            f.suptitle('input text TBD')
            axarr[0].imshow(input_spectrogram[:l].cpu().numpy().T)
            axarr[1].imshow(input_spectrogram_masked[:l].cpu().numpy().T)
            axarr[2].imshow(text_alignment[0][:l, :text_len].cpu().detach().numpy().T)
            axarr[3].imshow(pred_spectrogram[:l].cpu().detach().numpy().T)
            self.tb_logger.add_figure(
                f'{name_prefix}_{i+1}', f, global_step=self.global_step)

    def setup_training_data(self, cfg):
        if self.inference:
            return
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        if self.inference:
            return
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

        min_duration_frames = int(ffts_per_sec * self.min_mask_duration)
        max_duration_frames = int(ffts_per_sec * self.max_mask_duration)

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
    """See SpeechPainter paper for this"""
    input_width = 32

    def __init__(self, num_mels):
        super().__init__()
        input_width = self.input_width
        starting_filters = 2 ** 5
        # starting_filters = 2

        self.layers = [
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=starting_filters,
                kernel_size=(3, 3),
                padding='same'
            ),
            ConvUnit(
                (starting_filters, input_width, num_mels),
                starting_filters, 1, 1
            ),
            ConvUnit(
                (starting_filters * 2, input_width, num_mels),
                starting_filters * 2, 2, 2
            ),
            ConvUnit(
                (starting_filters * 4, input_width // 2, num_mels // 2),
                starting_filters * 4, 2, 2
            ),
            ConvUnit(
                (starting_filters * 8, input_width // 4, num_mels // 4),
                starting_filters * 8, 2, 2
            ),
            ConvUnit(
                (starting_filters * 16, input_width // 8, num_mels // 8),
                starting_filters * 16, 2, 2
            ),
            torch.nn.Conv2d(
                in_channels=starting_filters * 32,
                out_channels=1,
                kernel_size=(2, num_mels // 16),
            )
        ]
        # pytorch checks through named fields of a class to register the
        # modules as parameters of the model
        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)

    def forward(self, spectrogram_slices):
        activations = []
        h = spectrogram_slices.unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
            activations += [h]

        return h, activations


class ConvUnit(NeuralModule):
    def __init__(self, input_dims, c, s_t, s_f):
        """See SpeechPainter Paper"""
        super().__init__()
        in_channels, input_width, input_height = input_dims
        # first part:
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
            (c * 2, input_width // s_t, input_height // s_f))

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
            (c * 2, input_width // s_t, input_height // s_f))

    def forward(self, x):
        # part 1
        h = self.norm_1_1(self.conv_1_1(self.elu(x)))
        part_1 = self.norm_1_2(self.conv_1_2(self.elu(h)))

        # part 2
        part_2 = self.norm_2(self.conv_2(self.pooling(x)))
        return part_1 + part_2


def linear_rampup_with_warmup(step_nuber, warmup_steps, rampup_steps):
    adjusted_step_number = max(0, (step_nuber - warmup_steps))

    multiplier = min(1., adjusted_step_number / rampup_steps)
    return multiplier


def random_slice_if_longer_than(
    mels,
    spec_len,
    max_mel_len_for_train,
):
    """Randomly trucate examples if they are too long

    example:
        imagine a batch with examples of length:
        [5, 25, 4]
        and we want to make sure no batch is longer than 10 samples

        This function will random truncate the 2nd element in the batch to be
        of length 10


    the slice_indices return value represents the indices used for each element
    in the batch. You can use gather_slices to truncate similarly sized

    """
    # if there's no need to slice
    if mels.shape[1] <= max_mel_len_for_train:
        return mels, spec_len, None

    device = mels.device
    wiggle_room = torch.maximum(
        spec_len - max_mel_len_for_train,
        torch.zeros_like(spec_len, device=device),
    ).float()
    # add an eps to values where wiggle_room is 0 so uniform distribution
    # doesn't break
    mask = torch.eq(wiggle_room, 0).float()
    wiggle_room += (mask * 1e-5)

    dist = torch.distributions.uniform.Uniform(
        low=torch.zeros_like(wiggle_room),
        high=wiggle_room
    )
    start_indices = dist.sample().int()

    slice_indices = torch.arange(
        max_mel_len_for_train,
        device=device
    ).repeat(len(mels), 1)
    slice_indices += start_indices.unsqueeze(1)

    mels = gather_slices(mels, slice_indices)

    spec_len = torch.minimum(
        spec_len,
        torch.ones_like(spec_len, device=device) * max_mel_len_for_train
    )

    return mels, spec_len, slice_indices


def gather_slices(x, slice_indices):
    if slice_indices is None:
        return x
    return torch.stack([elem[si] for elem, si in zip(x, slice_indices)])
