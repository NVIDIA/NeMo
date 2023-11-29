import math
from functools import partial
from omegaconf import DictConfig

import torch
from torch import nn, Tensor, einsum, IntTensor, FloatTensor, BoolTensor
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Tuple, Optional, List, Union
from naturalspeech2_pytorch.utils.tokenizer import Tokenizer
from naturalspeech2_pytorch.aligner import Aligner, ForwardSumLoss, BinLoss, maximum_path
from voicebox_pytorch.voicebox_pytorch import VoiceBox as _VB
from voicebox_pytorch.voicebox_pytorch import ConditionalFlowMatcherWrapper as _CFMWrapper
from voicebox_pytorch.voicebox_pytorch import DurationPredictor as _DP
from voicebox_pytorch.voicebox_pytorch import (
    exists,
    coin_flip,
    mask_from_frac_lengths,
    prob_mask_like,
    curtail_or_pad,
    is_probably_audio_from_shape,
    default,
    unpack_one,
    pack_one
)
from torchaudio.functional import resample
import torchode as to
from torchdiffeq import odeint
from einops import rearrange, repeat, reduce, pack, unpack

from voicebox_pytorch.voicebox_pytorch import AudioEncoderDecoder
from voicebox_pytorch.voicebox_pytorch import MelVoco as _MelVoco
from voicebox_pytorch.voicebox_pytorch import EncodecVoco

from nemo.utils import logging


def get_mask_from_lengths(lengths, max_len=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


class MelVoco(_MelVoco):
    @property
    def downsample_factor(self):
        return 1.
    


class DurationPredictor(_DP):
    """
        1. Fixing `self.forward_aligner()`.
            Affecting:
                - `self.forward()`
                    - `self.forward_with_cond_scale()`
                        - `ConditionalFlowMatcherWrapper.sample()`
                        - `ConditionalFlowMatcherWrapper.forward()`
        2. Fix `self.forward()#L823`
            - L823-826: only keep positive condition, since `self_attn_mask` is ensured in L781-782
            - L828: `mask` seems to be corrected into `loss_mask`, while `loss_mask` is declared in L824, so simply remove the if statement
            - L841: `should_align` originally refers to the L818 assertion, therefore should be removed
    """
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        num_phoneme_tokens: Optional[int] = None,
        dim_phoneme_emb = 512,
        dim = 512,
        depth = 10,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_qk_norm = True,
        ff_dropout = 0.,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout=0,
        attn_flash = False,
        p_drop_prob = 0.2, # p_drop in paper
        frac_lengths_mask: List[float] = [0.1, 1.],
        aligner_kwargs: dict | DictConfig = dict(dim_in = 80, attn_channels = 80)
    ):
        super().__init__(
            audio_enc_dec=None,
            tokenizer=tokenizer,
            num_phoneme_tokens=num_phoneme_tokens,
            dim_phoneme_emb=dim_phoneme_emb,
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            attn_qk_norm=attn_qk_norm,
            ff_dropout=ff_dropout,
            conv_pos_embed_kernel_size=conv_pos_embed_kernel_size,
            conv_pos_embed_groups=conv_pos_embed_groups,
            attn_dropout=attn_dropout,
            attn_flash=attn_flash,
            p_drop_prob=p_drop_prob,
            frac_lengths_mask=tuple(frac_lengths_mask),
            aligner_kwargs=dict(aligner_kwargs)
        )
        # duration cond to emb
        self.proj_in = nn.Linear(1, dim)

    @torch.inference_mode()
    @beartype
    def forward_with_cond_scale(
        self,
        cond = None,
        texts: Optional[List[str]] = None,
        phoneme_ids = None,
        cond_scale = 1.,
        return_aligned_phoneme_ids = False,
        **kwargs
    ):
        """ Fix phoneme_ids device.

        Args:
            - cond: duration condition (ground truth duration)
            - texts
            - phoneme_ids
            - cond_scale: interpolate factor between cond/uncond
            - return_aligned_phoneme_ids
        """
        if exists(texts):
            phoneme_ids = self.tokenizer.texts_to_tensor_ids(texts).to(self.device)

        forward_kwargs = dict(
            return_aligned_phoneme_ids = False,
            phoneme_ids = phoneme_ids
        )

        durations = self.forward(cond_drop_prob = 0., **forward_kwargs, **kwargs)

        if cond_scale == 1.:
            if not return_aligned_phoneme_ids:
                return durations

            return durations, self.align_phoneme_ids_with_durations(phoneme_ids, durations)

        null_durations = self.forward(cond_drop_prob = 1., **forward_kwargs, **kwargs)
        scaled_durations = null_durations + (durations - null_durations) * cond_scale

        if not return_aligned_phoneme_ids:
            return scaled_durations

        return scaled_durations, self.align_phoneme_ids_with_durations(phoneme_ids, scaled_durations)


    @beartype
    def forward_aligner(
        self,
        x: FloatTensor,     # (b, tx, c)
        x_mask: IntTensor,  # (b, 1, tx)
        y: FloatTensor,     # (b, ty, c)
        y_mask: IntTensor   # (b, 1, ty)
    ) -> Tuple[
        FloatTensor,        # alignment_hard: (b, tx)
        FloatTensor,        # alignment_soft: (b, tx, ty)
        FloatTensor,        # alignment_logprob: (b, 1, ty, tx)
        BoolTensor          # alignment_mas: (b, tx, ty)
    ]:
        attn_mask = rearrange(x_mask, 'b 1 tx -> b 1 tx 1') * rearrange(y_mask, 'b 1 ty -> b 1 1 ty')
        alignment_soft, alignment_logprob = self.aligner.aligner(
            rearrange(y, 'b ty c -> b c ty'), rearrange(x, 'b tx c -> b c tx'), x_mask
        )

        assert not torch.isnan(alignment_soft).any()

        alignment_mas = maximum_path(
            rearrange(alignment_soft, 'b 1 ty tx -> b tx ty').contiguous(),
            rearrange(attn_mask, 'b 1 tx ty -> b tx ty').contiguous()
        )

        alignment_hard = torch.sum(alignment_mas, -1).float()
        alignment_soft = rearrange(alignment_soft, 'b 1 ty tx -> b tx ty')
        return alignment_hard, alignment_soft, alignment_logprob, alignment_mas

    @beartype
    def forward_aligner_from_text_mel(
        self,
        texts = None,
        phoneme_ids: IntTensor = None,  # (b, tx)
        mel: FloatTensor = None,        # (b, ty, c)
        mel_mask: FloatTensor = None    # (b, 1, ty)
    ) -> Tuple[
        FloatTensor,        # alignment_hard: (b, tx)
        FloatTensor,        # alignment_soft: (b, tx, ty)
        FloatTensor,        # alignment_logprob: (b, 1, ty, tx)
        BoolTensor          # alignment_mas: (b, tx, ty)
    ]:
        # text to phonemes, if tokenizer is given

        if not exists(phoneme_ids):
            assert exists(self.tokenizer) and exists(texts)
            phoneme_ids = self.tokenizer.texts_to_tensor_ids(texts).to(self.device)

        # phoneme id of -1 is padding

        phoneme_mask = phoneme_ids != -1
        phoneme_len = phoneme_mask.sum(-1)
        phoneme_mask =  rearrange(phoneme_mask, 'b t -> b 1 t')

        phoneme_ids = phoneme_ids.clamp(min = 0)

        # get phoneme embeddings

        phoneme_emb = self.to_phoneme_emb(phoneme_ids)

        return self.forward_aligner(phoneme_emb, phoneme_mask, mel, mel_mask)


    @beartype
    def forward(
        self,
        *,
        cond,
        texts: Optional[List[str]] = None,
        phoneme_ids = None,
        phoneme_len = None,
        phoneme_mask = None,
        cond_drop_prob = 0.,
        target = None,
        cond_mask = None,
        mel = None,
        mel_len = None,
        mel_mask = None,
        self_attn_mask = None,
        return_aligned_phoneme_ids = False,
        calculate_cond = False
    ):
        """ Allow passing `cond=None` while actually requires cond by setting `calculate_cond=True`
        """
        # text to phonemes, if tokenizer is given

        if not exists(phoneme_ids):
            assert exists(self.tokenizer) and exists(texts)
            phoneme_ids = self.tokenizer.texts_to_tensor_ids(texts).to(self.device)

        batch, seq_len = phoneme_ids.shape
        
        # phoneme id of -1 is padding

        if not exists(self_attn_mask):
            self_attn_mask = phoneme_ids != -1
        if not exists(phoneme_len):
            phoneme_len = self_attn_mask.sum(-1)
        if not exists(phoneme_mask):
            phoneme_mask =  rearrange(self_attn_mask, 'b t -> b 1 t')

        phoneme_ids = phoneme_ids.clamp(min = 0)

        # get phoneme embeddings

        phoneme_emb = self.to_phoneme_emb(phoneme_ids)

        # aligner
        # use alignment_hard to oversample phonemes
        # Duration Predictor should predict the duration of unmasked phonemes where target is masked alignment_hard

        assert all([exists(el) for el in (phoneme_len, mel_len, phoneme_mask, mel_mask)]), 'need to pass phoneme_len, mel_len, phoneme_mask, mel_mask, to train duration predictor module'

        alignment_hard, _, alignment_logprob, _ = self.forward_aligner(phoneme_emb, phoneme_mask, mel, mel_mask)
        target = alignment_hard

        # create dummy cond when not given, become purely unconditional regression model

        if not exists(cond):
            if calculate_cond:
                cond = alignment_hard
            else:
                cond = torch.zeros_like(phoneme_ids)
                cond_drop_prob = 1

        cond = self.proj_in(cond)

        # construct mask if not given

        if not exists(cond_mask):
            if coin_flip():
                frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask)
                cond_mask = mask_from_frac_lengths(seq_len, frac_lengths)
            else:
                cond_mask = prob_mask_like((batch, seq_len), self.p_drop_prob, self.device)

        cond = cond * rearrange(~cond_mask, '... -> ... 1')

        # classifier free guidance

        if cond_drop_prob > 0.:
            cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, cond.device)

            cond = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'),
                self.null_cond,
                cond
            )

        # force condition to be same length as input phonemes

        cond = curtail_or_pad(cond, phoneme_ids.shape[-1])

        # combine audio, phoneme, conditioning

        embed = torch.cat((phoneme_emb, cond), dim = -1)
        x = self.to_embed(embed)

        x = self.conv_embed(x) + x

        x = self.transformer(
            x,
            mask = self_attn_mask
        )

        durations = self.to_pred(x)

        if not self.training:
            if not return_aligned_phoneme_ids:
                return durations

            return durations, self.align_phoneme_ids_with_durations(phoneme_ids, durations)

        loss_mask = cond_mask & self_attn_mask

        loss = F.l1_loss(durations, target, reduction = 'none')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den
        loss = loss.mean()
        losses = {"d_pred_loss": loss.detach()}

        #aligner loss

        align_loss = self.align_loss(alignment_logprob, phoneme_len, mel_len)
        loss = loss + align_loss
        losses["align_loss"] = align_loss.detach()

        if not return_aligned_phoneme_ids:
            return loss
        return loss, losses, self.align_phoneme_ids_with_durations(phoneme_ids=phoneme_ids, durations=target)


class VoiceBox(_VB):
    """ Nothing to fix currently. Add some docs.
    """
    def __init__(
        self,
        *_args,
        num_cond_tokens = None,
        audio_enc_dec: Optional[AudioEncoderDecoder] = None,
        dim_in = None,
        dim_cond_emb = 1024,
        dim = 1024,
        depth = 24,
        dim_head = 64,
        heads = 16,
        ff_mult = 4,
        ff_dropout = 0.,
        time_hidden_dim = None,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout = 0,
        attn_flash = False,
        attn_qk_norm = True,
        num_register_tokens = 16,
        p_drop_prob = 0.3, # p_drop in paper
        frac_lengths_mask: Tuple[float, float] = (0.7, 1.),
        condition_on_text = True
    ):
        """
        Input related args:

            - audio_enc_dec: Optional[AudioEncoderDecoder] = None, for EnCodecVoco or MelVoco
        
            - dim_cond_emb = 1024,
            
            - dim = 1024,

                - dim_in = None,

                        have to be None or equal to dim.

                        should be deprecated and replaced with dim
            
            - time_hidden_dim = None, for time step embedding

        ConvPositionEmbed args

            - conv_pos_embed_kernel_size = 31,
        
            - conv_pos_embed_groups = None,

        Transformer specific args

            - depth = 24,
        
            - dim_head = 64,
            
            - heads = 16,
            
            - ff_mult = 4,
            
            - ff_dropout = 0.,
            
            - attn_dropout = 0,
            
            - attn_flash = False,
            
            - attn_qk_norm = True,
            
            - num_register_tokens = 16,

        Conditional training args

            - num_cond_tokens = None,
        
            - condition_on_text = True
            
            - p_drop_prob = 0.3
            
                    p_drop in paper
            
            - frac_lengths_mask: Tuple[float, float] = (0.7, 1.),
        """
        super().__init__(
            *_args,
            num_cond_tokens=num_cond_tokens,
            audio_enc_dec=audio_enc_dec,
            dim_in=dim_in,
            dim_cond_emb=dim_cond_emb,
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
            time_hidden_dim=time_hidden_dim,
            conv_pos_embed_kernel_size=conv_pos_embed_kernel_size,
            conv_pos_embed_groups=conv_pos_embed_groups,
            attn_dropout=attn_dropout,
            attn_flash=attn_flash,
            attn_qk_norm=attn_qk_norm,
            num_register_tokens=num_register_tokens,
            p_drop_prob=p_drop_prob,
            frac_lengths_mask=frac_lengths_mask,
            condition_on_text=condition_on_text
        )

    def forward(self, *args, **kwargs):
        """
        Parameters:
            x: x_t
            times: t
            cond_token_ids: y (expended phonemes)
                Phonemes should have already expended, or else treated as SemanticTokens (w2v-bert/hubert tokens) and interpolate
            self_attn_mask
            cond_drop_prob
            target
            cond: optional. `x` (target spectrogram)
            cond_mask: optional.
        """
        return super().forward(*args, **kwargs)
    

class ConditionalFlowMatcherWrapper(_CFMWrapper):
    """ Deal with `self.forward()` duration prediction and aligner.
    """
    @beartype
    def __init__(
        self,
        voicebox: VoiceBox,
        duration_predictor: Optional[DurationPredictor] = None,
        sigma = 0.,
        ode_atol = 1e-5,
        ode_rtol = 1e-5,
        ode_step_size = 0.0625,
        use_torchode = False,
        torchdiffeq_ode_method = 'midpoint',   # use midpoint for torchdiffeq, as in paper
        torchode_method_klass = to.Tsit5,      # use tsit5 for torchode, as torchode does not have midpoint (recommended by Bryan @b-chiang)
        cond_drop_prob = 0.
    ):
        """
        Basic Args
            - voicebox: Voicebox,

        Input related args

            - duration_predictor: Optional[DurationPredictor] = None,

                    Take phoneme sequence as input

        ODE args

            - ode_atol = 1e-5,

            - ode_rtol = 1e-5,

            - ode_step_size = 0.0625,

            - use_torchode = False,

            - torchdiffeq_ode_method = 'midpoint',

                use midpoint for torchdiffeq, as in paper

            - torchode_method_klass = to.Tsit5,

                use tsit5 for torchode, as torchode does not have midpoint (recommended by Bryan @b-chiang)

        Voicebox forward args

            - sigma = 0.,

            - cond_drop_prob = 0.
        """
        super().__init__(
            voicebox=voicebox,
            text_to_semantic=None,
            duration_predictor=duration_predictor,
            sigma=sigma,
            ode_atol=ode_atol,
            ode_rtol=ode_rtol,
            ode_step_size=ode_step_size,
            use_torchode=use_torchode,
            torchdiffeq_ode_method=torchdiffeq_ode_method,
            torchode_method_klass=torchode_method_klass,
            cond_drop_prob=cond_drop_prob
        )
        self.duration_predictor: DurationPredictor

        
    @torch.inference_mode()
    def sample(
        self,
        *_args,
        dp_cond = None,
        cond = None,
        texts: Optional[List[str]] = None,
        text_token_ids: Optional[Tensor] = None,
        semantic_token_ids: Optional[Tensor] = None,
        phoneme_ids: Optional[Tensor] = None,
        cond_mask = None,
        steps = 3,
        cond_scale = 1.,
        decode_to_audio = True,
        max_semantic_token_ids = 2048,
        spec_decode = False,
        spec_decode_gamma = 5 # could be higher, since speech is probably easier than text, needs to be tested
    ):
        """
        Handle slf_attn_mask (cond_mask)
        
        Args:
            - dp_cond: duration predictor condition (duration ground truth)
            - cond: reference audio conditioning for voicebox
            - texts:
            - text_token_ids:
            - semantic_token_ids:
            - phoneme_ids:

            - cond_mask: condition masking -> context
            - steps: ODE steps
            - cond_scale: classifier-free guidance
        """
        # take care of condition as raw audio

        cond_is_raw_audio = is_probably_audio_from_shape(cond)

        if cond_is_raw_audio:
            assert exists(self.voicebox.audio_enc_dec)

            self.voicebox.audio_enc_dec.eval()
            cond = self.voicebox.audio_enc_dec.encode(cond)

        # setup text conditioning, either coming from duration model (as phoneme ids)
        # for coming from text-to-semantic module from spear-tts paper, as (semantic ids)

        num_cond_inputs = sum([*map(exists, (texts, text_token_ids, semantic_token_ids, phoneme_ids))])
        assert num_cond_inputs <= 1

        self_attn_mask = None
        cond_token_ids = None

        if self.condition_on_text:
            assert not (exists(self.text_to_semantic) or exists(semantic_token_ids))
            assert exists(self.duration_predictor)

            self.duration_predictor.eval()

            durations, aligned_phoneme_ids = self.duration_predictor.forward_with_cond_scale(
                cond=dp_cond,
                texts=texts,
                phoneme_ids=phoneme_ids,
                cond_scale=1,
                return_aligned_phoneme_ids=True
            )

            cond_token_ids = aligned_phoneme_ids

            cond_tokens_seq_len = cond_token_ids.shape[-1]

            if exists(cond):
                assert not exists(self.text_to_semantic)
                assert exists(self.duration_predictor)
                cond_target_length = cond_tokens_seq_len

                cond = curtail_or_pad(cond, cond_target_length)
                self_attn_mask = curtail_or_pad(torch.ones_like(cond, dtype=torch.bool), cond_target_length)
            else:
                cond = torch.zeros((cond_token_ids.shape[0], cond_target_length, self.dim_cond_emb), device = self.device)
        else:
            assert num_cond_inputs == 0, 'no conditioning inputs should be given if not conditioning on text'

        shape = cond.shape
        batch = shape[0]

        # neural ode

        self.voicebox.eval()

        def fn(t, x, *, packed_shape = None):
            if exists(packed_shape):
                x = unpack_one(x, packed_shape, 'b *')

            out = self.voicebox.forward_with_cond_scale(
                x,
                times = t,
                cond_token_ids = cond_token_ids,
                cond = cond,
                cond_scale = cond_scale,
                cond_mask = cond_mask,
                self_attn_mask = self_attn_mask
            )

            if exists(packed_shape):
                out = rearrange(out, 'b ... -> b (...)')

            return out

        y0 = torch.randn_like(cond)
        t = torch.linspace(0, 1, steps, device = self.device)

        if not self.use_torchode:
            logging.debug('sampling with torchdiffeq')

            trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
            sampled = trajectory[-1]
        else:
            logging.debug('sampling with torchode')

            t = repeat(t, 'n -> b n', b = batch)
            y0, packed_shape = pack_one(y0, 'b *')

            fn = partial(fn, packed_shape = packed_shape)

            term = to.ODETerm(fn)
            step_method = self.torchode_method_klass(term = term)

            step_size_controller = to.IntegralController(
                atol = self.odeint_kwargs['atol'],
                rtol = self.odeint_kwargs['rtol'],
                term = term
            )

            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            jit_solver = torch.compile(solver)

            init_value = to.InitialValueProblem(y0 = y0, t_eval = t)

            sol = jit_solver.solve(init_value)

            sampled = sol.ys[:, -1]
            sampled = unpack_one(sampled, packed_shape, 'b *')

        if not decode_to_audio or not exists(self.voicebox.audio_enc_dec):
            return sampled

        return self.voicebox.audio_enc_dec.decode(sampled)

    def forward(
        self,
        x1,
        *,
        mask = None,
        phoneme_ids = None,
        phoneme_len = None,
        dp_cond = None,
        cond = None,
        cond_mask = None,
        input_sampling_rate = None # will assume it to be the same as the audio encoder decoder sampling rate, if not given. if given, will resample
    ):
        """TODO: Deal with phoneme duration alignment and expansion

        Args

            - x1,

                    input audio    

            - mask = None,

                    voicebox self_attn_mask

            - semantic_token_ids = None,

                    pass in if using semantic tokens as text input

            - phoneme_ids = None,

                    pass in if using phoneme sequence as text input

            - cond = None,

                    for context audio, could set to x1 or None

            - cond_mask = None,

                    masking context audio

                    could pass None and let Voicebox generate for you

            - input_sampling_rate = None

                    resample if given & != vocoder sampling rate
        """

        batch, seq_len, dtype, σ = *x1.shape[:2], x1.dtype, self.sigma

        # if raw audio is given, convert if audio encoder / decoder was passed in

        input_is_raw_audio, cond_is_raw_audio = map(is_probably_audio_from_shape, (x1, cond))

        if input_is_raw_audio:
            raw_audio = x1

        if any([input_is_raw_audio, cond_is_raw_audio]):
            assert exists(self.voicebox.audio_enc_dec), 'audio_enc_dec must be set on VoiceBox to train directly on raw audio'

            audio_enc_dec_sampling_rate = self.voicebox.audio_enc_dec.sampling_rate
            input_sampling_rate = default(input_sampling_rate, audio_enc_dec_sampling_rate)

            with torch.no_grad():
                self.voicebox.audio_enc_dec.eval()

                if input_is_raw_audio:
                    x1 = resample(x1, input_sampling_rate, audio_enc_dec_sampling_rate)
                    x1 = self.voicebox.audio_enc_dec.encode(x1)

                if exists(cond) and cond_is_raw_audio:
                    cond = resample(cond, input_sampling_rate, audio_enc_dec_sampling_rate)
                    cond = self.voicebox.audio_enc_dec.encode(cond)

        # setup text conditioning, either coming from duration model (as phoneme ids)
        # or from text-to-semantic module, semantic ids encoded with wav2vec (hubert usually)

        assert self.condition_on_text or not exists(phoneme_ids), 'phoneme ids should not be passed in if not conditioning on text'

        cond_token_ids = None




        # NOTE: work in progress
        # TODO: allow duration predictor using different audio_enc_dec
        assert exists(phoneme_ids)

        self_attn_mask = None
        cond_token_ids = None

        # handle downsample audio_mask
        if input_is_raw_audio:
            audio_mask = resample(mask, input_sampling_rate, audio_enc_dec_sampling_rate) >= 0.5
            audio_len = audio_mask.sum(-1)
            mel_len = audio_len / audio_enc_dec_sampling_rate / self.voicebox.audio_enc_dec.downsample_factor
            mel_mask = get_mask_from_lengths(mel_len)


        if self.condition_on_text:
            assert not exists(self.text_to_semantic)
            assert exists(self.duration_predictor)

            self.duration_predictor.train()
            phoneme_mask = phoneme_ids != -1
            if not exists(phoneme_len):
                phoneme_len = phoneme_mask.sum(-1)

            dp_loss, dp_losses, aligned_phoneme_ids = self.duration_predictor.forward(
                cond=dp_cond,               # might be None
                texts=None,                 # converted to phoneme_ids by dataset
                phoneme_ids=phoneme_ids,
                phoneme_len=phoneme_len,
                phoneme_mask=None,          # would be calculated within
                cond_drop_prob=0.2,
                target=None,                # unused
                cond_mask=None,             # would be generated within
                mel=x1,                     # TODO: not assuming DP using same audio_enc_dec with VB
                mel_len=mel_len,
                mel_mask=mel_mask,
                self_attn_mask=None,        # would be calculated within
                return_aligned_phoneme_ids=True,
                calculate_cond=True
            )

            cond_token_ids = aligned_phoneme_ids

            cond_tokens_seq_len = cond_token_ids.shape[-1]

            if exists(cond):
                assert not exists(self.text_to_semantic)
                assert exists(self.duration_predictor)
                cond_target_length = cond_tokens_seq_len

                cond = curtail_or_pad(cond, cond_target_length)
                self_attn_mask = curtail_or_pad(torch.ones_like(cond, dtype=torch.bool), cond_target_length)
            else:
                cond = torch.zeros((cond_token_ids.shape[0], cond_target_length, self.dim_cond_emb), device = self.device)
        else:
            assert not exists(phoneme_ids), 'no conditioning inputs should be given if not conditioning on text'


        # main conditional flow logic is below

        # x0 is gaussian noise

        x0 = torch.randn_like(x1)

        # random times

        times = torch.rand((batch,), dtype = dtype, device = self.device)
        t = rearrange(times, 'b -> b 1 1')

        # sample xt (w in the paper)

        w = (1 - (1 - σ) * t) * x0 + t * x1

        flow = x1 - (1 - σ) * x0

        # predict

        self.voicebox.train()

        loss = self.voicebox(
            w,
            cond = cond,
            cond_mask = cond_mask,
            times = times,
            target = flow,
            self_attn_mask = self_attn_mask,
            cond_token_ids = cond_token_ids,
            cond_drop_prob = self.cond_drop_prob
        )

        losses = {}
        if self.condition_on_text:
            losses.update(dp_losses)
        losses['vb_loss'] = loss
        loss += dp_loss

        return loss, losses

        # return super().forward(x1=x1, mask=mask, semantic_token_ids=semantic_token_ids,phoneme_ids=phoneme_ids, cond=cond, cond_mask=cond_mask, input_sampling_rate=input_sampling_rate)