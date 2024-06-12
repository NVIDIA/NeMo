import math
from functools import partial
from omegaconf import DictConfig

import torch
from torch import nn, Tensor, einsum, IntTensor, FloatTensor, BoolTensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from beartype import beartype
from beartype.typing import Callable, Tuple, Optional, List, Union
from naturalspeech2_pytorch.utils.tokenizer import Tokenizer, _phonemes
from naturalspeech2_pytorch.aligner import Aligner as _Aligner
from naturalspeech2_pytorch.aligner import maximum_path
from naturalspeech2_pytorch.utils.cleaner import TextProcessor
from voicebox_pytorch.voicebox_pytorch import VoiceBox as _VB
from voicebox_pytorch.voicebox_pytorch import ConditionalFlowMatcherWrapper as _CFMWrapper
from voicebox_pytorch.voicebox_pytorch import DurationPredictor as _DP
from voicebox_pytorch.voicebox_pytorch import (
    is_probably_audio_from_shape,
    Transformer,
    Rearrange,
)
import torchaudio.transforms as T
from torchaudio.functional import resample
import torchode as to
from torchdiffeq import odeint
from einops import rearrange, repeat, reduce, pack, unpack

from voicebox_pytorch.voicebox_pytorch import AudioEncoderDecoder
from voicebox_pytorch.voicebox_pytorch import MelVoco as _MelVoco
from voicebox_pytorch.voicebox_pytorch import EncodecVoco as _EncodecVoco
import dac

from pytorch_lightning import LightningModule
from nemo.utils import logging
from nemo.collections.tts.models.aligner import AlignerModel
from nemo.collections.asr.modules.audio_preprocessing import AudioPreprocessor
# from nemo.collections.tts.parts.utils.helpers import binarize_attention
from nemo.collections.tts.parts.utils.helpers import binarize_attention_parallel as binarize_attention
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer, EnglishPhonemesTokenizer
from nemo.collections.tts.modules.voicebox_utils import (
    exists,
    default,
    coin_flip,
    pack_one,
    unpack_one,
    prob_mask_like,
    reduce_masks_with_and,
    interpolate_1d,
    curtail_or_pad,
    mask_from_start_end_indices,
    mask_from_frac_lengths,
    generate_mask_from_repeats,
    LearnedSinusoidalPosEmb,
    ConvPositionEmbed,
)


class MFAEnglishPhonemeTokenizer(Tokenizer):
    MFA_arpa_phone_set = ["PAD", "sil", "spn", "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2", "AO", "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY", "EY0", "EY1", "EY2", "F", "G", "HH", "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW", "OW0", "OW1", "OW2", "OY", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"]
    word_postfix = ["_S", "_B", "_I", "_E"]

    def __init__(
        self,
        vocab = MFA_arpa_phone_set,
        add_blank: bool = False,
        use_eos_bos = False,
        pad_id = 0,
        use_word_postfix = False,
        **kwargs
    ):
        self.add_blank = add_blank
        self.use_eos_bos = use_eos_bos
        self.pad_id = pad_id
        self.use_word_postfix = use_word_postfix

        # vocab.pop(vocab.index("PAD"))
        vocab = [phn for phn in vocab if phn != "PAD"]
        if self.use_word_postfix:
            vocab = [phn+_p for phn in vocab for _p in self.word_postfix]
        vocab.insert(0, "PAD")

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.phn_to_id = {phn: idx for idx, phn in enumerate(self.vocab)}
        self.id_to_phn = {idx: phn for idx, phn in enumerate(self.vocab)}

        self.not_found_phonemes = []

    def encode(self, text: List[str]) -> List[int]:
        """Encodes a string of text as a sequence of IDs."""
        token_ids = []
        for phn in text:
            if phn == "":
                if self.use_word_postfix:
                    phn = "sil_S"
                else:
                    phn = "sil"
            try:
                idx = self.phn_to_id[phn]
                token_ids.append(idx)
            except KeyError:
                # discard but store not found phonemes
                if phn not in self.not_found_phonemes:
                    self.not_found_phonemes.append(phn)
                    print(text)
                    print(f" [!] Character {phn} not found in the vocabulary. Discarding it.")
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decodes a sequence of IDs to a string of text."""
        tokens = []
        for token_id in token_ids:
            tokens.append(self.id_to_phn[token_id])
        text = ' '.join(tokens)
        return text

    def text_to_ids(
        self,
        text: List[str],
        language: str = None
    ) -> Tuple[List[int],]:
        return self.encode(text),

    def texts_to_tensor_ids(self, texts: List[str], language: str = None) -> Tensor:
        all_ids = []

        for text in texts:
            ids, *_ = self.text_to_ids(text, language = language)
            all_ids.append(torch.tensor(ids))

        return pad_sequence(all_ids, batch_first = True, padding_value = self.pad_id)


class MelVoco(_MelVoco, LightningModule):
    def __init__(self, *args, normalize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze()
        self.normalize = normalize
        self.global_mean = -5.8843
        self.global_std = 2.2615

    def encode(self, audio):
        mel = self.vocos.feature_extractor(audio)

        mel = rearrange(mel, 'b d n -> b n d')
        if self.normalize:
            mel = (mel - self.global_mean) / self.global_std
        return mel

    def decode(self, mel):
        mel = rearrange(mel, 'b n d -> b d n')
        if self.normalize:
            mel = (mel * self.global_std) + self.global_mean

        return self.vocos.decode(mel)


class EncodecVoco(_EncodecVoco, LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze()


class DACVoco(AudioEncoderDecoder, LightningModule):
    def __init__(
        self,
        *,
        sampling_rate = 16000,
        pretrained_path = '16khz',
        bandwidth_id = None,
        factorized_latent = False,
        return_code = False,
        preq_ce = False,
        ce_weights = None,
        normalize = False,
    ):
        super().__init__()
        if pretrained_path in ["44khz", "24khz", "16khz"]:
            model_path = dac.utils.download(model_type=pretrained_path)
        else:
            model_path = pretrained_path
        try:
            self.model = dac.DAC.load(model_path)
        except:
            model_path = dac.utils.download(model_type="16khz")
            self.model = dac.DAC.load(model_path)
        self.sampling_rate = sampling_rate
        assert self.sampling_rate == self.model.sample_rate

        bandwidth_id = self.model.n_codebooks if not bandwidth_id else bandwidth_id
        self.register_buffer('bandwidth_id', torch.tensor([bandwidth_id]))

        self.normalize = normalize

        # factorize to 8-dim per code * bandwidth_id codebooks
        self.register_buffer('factorized_latent', torch.BoolTensor([factorized_latent]))

        # use code tokens
        self.register_buffer('return_code', torch.BoolTensor([return_code]))

        # pre-quantize feature + cross-entropy loss
        self.register_buffer('preq_ce', torch.BoolTensor([preq_ce]))
        self.ce_weights = None if ce_weights is None else torch.tensor(ce_weights[:self.bandwidth_id])
        if self.normalize:
            self.global_mean = -0.0350
            self.global_std = 2.6780

        self.freeze()

    @property
    def downsample_factor(self):
        return self.model.hop_length

    @property
    def latent_dim(self):
        if self.factorized_latent or self.return_code:
            return self.model.codebook_dim * self.model.n_codebooks
        else:
            return self.model.latent_dim

    @property
    def masked_latent_dim(self):
        if self.factorized_latent or self.return_code:
            return self.model.codebook_dim * self.bandwidth_id
        else:
            return self.model.latent_dim

    @torch.no_grad()
    def encode(self, audio):
        audio = rearrange(audio, 'b t -> b 1 t')
        audio = self.model.preprocess(audio, self.sampling_rate)

        if self.factorized_latent:
            z, codes, latents, _, _ = self.model.encode(audio)
            latents[:, self.masked_latent_dim:, :] = 0
            return rearrange(latents, 'b d n -> b n d')

        elif self.return_code:
            z, codes, latents, _, _ = self.model.encode(audio)
            codes[:, self.bandwidth_id:, :] = 0
            return rearrange(codes, 'b d n -> b n d')

        elif self.preq_ce:
            z = self.model.encoder(audio)
            if self.normalize:
                z = (z - self.global_mean) / self.global_std
            return rearrange(z, 'b d n -> b n d')

        else:
            z, codes, latents, _, _ = self.model.encode(
                audio, self.bandwidth_id
            )
            return rearrange(z, 'b d n -> b n d')

    def decode(self, latents):
        latents = rearrange(latents, 'b n d -> b d n')

        if self.factorized_latent:
            latents = latents[:, :self.masked_latent_dim, :]
            z_q, z_p, codes = self.model.quantizer.from_latents(latents)
        elif self.return_code:
            codes = latents[:, :self.bandwidth_id, :]
            z_q, z_p, codes = self.model.quantizer.from_codes(codes)
        elif self.preq_ce:
            z = latents
            if self.normalize:
                z = (z * self.global_std) + self.global_mean
            z_q, codes, latents, _, _ = self.model.quantizer(z, self.bandwidth_id)
        else:
            z_q = latents

        audio = self.model.decode(z_q)
        audio = rearrange(audio, 'b 1 t -> b t')

        return audio

    def cross_entropy_loss(self, z_pred, z):
        """
        Args:
            z_pred: predicted pre-quantize z
            z: ground truth pre-quantize z
        """
        z_pred = rearrange(z_pred, 'b n d -> b d n')
        z = rearrange(z, 'b n d -> b d n')

        if self.normalize:
            z_pred = (z_pred * self.global_std) + self.global_mean
            z = (z * self.global_std) + self.global_mean

        with torch.no_grad():
            _, codes, _, _, _ = self.model.quantizer.forward(z, self.bandwidth_id)

        z_q = 0
        residual = z_pred
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []
        z_e = []
        distances = []

        n_quantizers = self.bandwidth_id

        for i, quantizer in enumerate(self.model.quantizer.quantizers):
            if i >= n_quantizers:
                break

            # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
            z_e_i = quantizer.in_proj(residual)

            encodings = rearrange(z_e_i, "b d t -> (b t) d")
            codebook = quantizer.codebook.weight  # codebook: (N x D)

            # L2 normalize encodings and codebook (ViT-VQGAN)
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

            # Compute euclidean distance with codebook -> ((b t) N)
            dist = (
                encodings.pow(2).sum(1, keepdim=True)
                - 2 * encodings @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t()
            )

            indices_i = codes[:, i]
            z_q_i = quantizer.decode_code(indices_i)
            z_q_i = quantizer.out_proj(z_q_i)

            z_q = z_q + z_q_i
            residual = residual - z_q_i

            z_e.append(z_e_i)
            distances.append(dist)

        distances = -torch.stack(distances, -1)
        ce_loss = F.cross_entropy(
            rearrange(distances, '(b t) c i -> b c t i', b = z_pred.shape[0]),
            rearrange(codes, 'b i t -> b t i')[:, :, :n_quantizers],
            ignore_index = -1,
            reduction='none',
        )
        if self.ce_weights is not None:
            self.ce_weights = self.ce_weights.to(self.device)
            ce_loss = ce_loss * self.ce_weights

        return ce_loss


class Aligner(_Aligner):

    # def align_phoneme_ids_with_durations(self, phoneme_ids, durations):
    #     repeat_mask = generate_mask_from_repeats(durations.clamp(min = 0))
    #     aligned_phoneme_ids = einsum('b i, b i j -> b j', phoneme_ids.float(), repeat_mask.float()).long()
    #     return aligned_phoneme_ids

    def forward(self, x, x_mask, y, y_mask):
        alignment_soft, alignment_logprob = self.aligner(rearrange(y, 'b ty d -> b d ty'), rearrange(x, 'b tx d -> b d tx'), x_mask)

        x_mask = rearrange(x_mask, '... tx -> ... tx 1')
        y_mask = rearrange(y_mask, '... ty -> ... 1 ty')
        attn_mask = x_mask * y_mask
        attn_mask = rearrange(attn_mask, 'b 1 tx ty -> b tx ty')

        alignment_soft = rearrange(alignment_soft, 'b 1 ty tx -> b tx ty')
        alignment_mask = maximum_path(alignment_soft, attn_mask)

        alignment_hard = torch.sum(alignment_mask, -1).int()
        return alignment_hard, alignment_soft, alignment_logprob, alignment_mask


class DurationPredictor(_DP, LightningModule):
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
        *args,
        aligner_kwargs: Optional[dict | DictConfig] = None,
        **kwargs,
    ):
        kwargs["frac_lengths_mask"] = tuple(kwargs["frac_lengths_mask"])
        super().__init__(*args, **kwargs, aligner_kwargs={})

        if aligner_kwargs is not None:
            # if we are using mel spec with 80 channels, we need to set attn_channels to 80
            # dim_in assuming we have spec with 80 channels

            self.aligner = Aligner(dim_in = self.audio_enc_dec.latent_dim, dim_hidden = self.dim_phoneme_emb, **aligner_kwargs)

            for layer in self.aligner.modules():
                if isinstance(layer, nn.ReLU):
                    layer.inplace = False

            from naturalspeech2_pytorch.aligner import ForwardSumLoss, BinLoss
            self.align_loss = ForwardSumLoss()
            self.bin_loss = BinLoss()

    def align_phoneme_ids_with_durations(self, phoneme_ids, durations):
        repeat_mask = generate_mask_from_repeats(durations.clamp(min = 0))
        aligned_phoneme_ids = einsum('b i, b i j -> b j', phoneme_ids.float(), repeat_mask.float()).long()
        return aligned_phoneme_ids

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
        """
        Args:
            x: phone
            y: mel
        """
        return self.aligner(x, x_mask, y, y_mask)

    @torch.no_grad()
    def parse_dp_input(self, x1, mask, durations=None, phoneme_len=None, input_sampling_rate=None):
        assert exists(self.audio_enc_dec), 'audio_enc_dec must be set to train directly on raw audio'
        input_sampling_rate = default(input_sampling_rate, self.audio_enc_dec.sampling_rate)
        dp_inputs = {}

        input_is_raw_audio = is_probably_audio_from_shape(x1)
        if input_is_raw_audio and not isinstance(self, NeMoDurationPredictor):
            self.audio_enc_dec.eval()
            audio_enc_dec_sampling_rate = self.audio_enc_dec.sampling_rate

            mel = resample(x1, input_sampling_rate, audio_enc_dec_sampling_rate)
            mel = self.audio_enc_dec.encode(mel)
            
            audio_len = mask.sum(-1)
            mel_len = audio_len * mel.shape[1] // mask.shape[-1]
            mel_mask = get_mask_from_lengths(mel_len)
            mel_mask = rearrange(mel_mask, 'b t -> b 1 t')
            dp_inputs.update({
                "mel": mel,
                "mel_len": mel_len,
                "mel_mask": mel_mask
            })

            if durations is not None:
                cum_dur = torch.cumsum(durations, -1)
                dur_ratio = mel_len / cum_dur[:, -1]
                cum_dur = cum_dur * rearrange(dur_ratio, 'b -> b 1')
                cum_dur = torch.round(cum_dur)

                dp_cond = torch.zeros_like(cum_dur)
                dp_cond[:, 0] = cum_dur[:, 0]
                dp_cond[:, 1:] = cum_dur[:, 1:] - cum_dur[:, :-1]

                dp_inputs.update({
                    "dp_cond": dp_cond,
                    "cum_dur": cum_dur,
                })

        else:
            dp_inputs.update({
                "mel": x1,
                "mel_len": mask.sum(-1),
                "mel_mask": rearrange(mask, 'b t -> b 1 t')
            })

        assert exists(phoneme_len)
        phoneme_mask = get_mask_from_lengths(phoneme_len)
        dp_inputs.update({
            "phoneme_mask": phoneme_mask
        })

        return dp_inputs


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
        `cond` should be ground-truth duration instead of encoded audio
        """
        outputs = {}

        # text to phonemes, if tokenizer is given
        phoneme_ids = self.to_phoneme_ids(texts, phoneme_ids)
        
        # phoneme id of -1 is padding
        assert exists(self_attn_mask) and exists(phoneme_len) and exists(phoneme_mask)

        if phoneme_mask.ndim == 2:
            phoneme_mask =  rearrange(self_attn_mask, 'b t -> b 1 t')
        # phoneme_ids = phoneme_ids.clamp(min = 0)

        # get phoneme embeddings
        phoneme_emb = self.to_phoneme_emb(phoneme_ids)

        # aligner
        # use alignment_hard to oversample phonemes
        # Duration Predictor should predict the duration of unmasked phonemes where target is masked alignment_hard

        assert all([exists(el) for el in (phoneme_len, mel_len, phoneme_mask, mel_mask)]), 'need to pass phoneme_len, mel_len, phoneme_mask, mel_mask, to train duration predictor module'

        alignment_hard, alignment_soft, alignment_logprob, alignment_mas = self.forward_aligner(phoneme_emb, phoneme_mask, mel, mel_mask)
        target = alignment_hard
        outputs["target"] = target

        # create dummy cond when not given, become purely unconditional regression model

        if not exists(cond):
            if calculate_cond:
                cond = alignment_hard
            else:
                cond = torch.zeros_like(phoneme_ids)
                cond_drop_prob = 1

        cond = rearrange(cond, 'b t -> b t 1')
        cond = self.proj_in(cond)

        # construct mask if not given

        if not exists(cond_mask):
            batch, seq_len = phoneme_ids.shape
            cond_mask = self.create_cond_mask(batch=batch, seq_len=seq_len, training=self.training)

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
        outputs["durations"] = durations

        if not self.training:
            if return_aligned_phoneme_ids:
                aligned_phoneme_ids = self.align_phoneme_ids_with_durations(phoneme_ids, durations)
                outputs["aligned_phoneme_ids"] = aligned_phoneme_ids

            return outputs

        loss_mask = cond_mask & self_attn_mask

        loss = F.l1_loss(durations, target, reduction = 'none')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den
        loss = loss.mean()

        #aligner loss

        align_loss = self.align_loss(alignment_logprob, phoneme_len, mel_len)
        loss = loss + align_loss

        # bin_loss = self.bin_loss(alignment_mas, alignment_logprob, phoneme_len)
        alignment_soft = rearrange(alignment_soft, 'b tx ty -> b 1 ty tx')
        alignment_mas = rearrange(alignment_mas, 'b tx ty -> b 1 ty tx')
        bin_loss = self.bin_loss(hard_attention=alignment_mas, soft_attention=alignment_soft)
        loss = loss + bin_loss
        
        losses = {
            "dp": loss,
            "align": align_loss,
            "bin": bin_loss
        }

        if return_aligned_phoneme_ids:
            aligned_phoneme_ids = self.align_phoneme_ids_with_durations(phoneme_ids=phoneme_ids, durations=target)
            outputs["aligned_phoneme_ids"] = aligned_phoneme_ids

        return loss, losses, outputs


class NeMoDurationPredictor(DurationPredictor):
    def __init__(
        self,
        aligner: Aligner = None,
        audio_enc_dec: Optional[AudioPreprocessor] = None,
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
        **kwargs,
    ):
        """
        1. `cond` arg of `forward() should be ground-truth duration, therefore fix `self.proj_in` into nn.Linear(1,dim)
        """
        # audio_enc_dec = aligner.preprocessor
        num_phoneme_tokens = aligner.embed.num_embeddings
        dim_phoneme_emb = aligner.embed.embedding_dim

        super().__init__(
            audio_enc_dec=audio_enc_dec,
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
            frac_lengths_mask=frac_lengths_mask,
            aligner_kwargs=None,
            **kwargs
        )

        self.aligner: AlignerModel = aligner

        self.tokenizer = self.aligner.tokenizer

        # self.to_phoneme_emb = self.aligner.embed

        # aligner related
        from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss, BinLoss


    @torch.no_grad()
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
        """
        Args:
            x: phone
            y: mel
        """
        audio = resample(y, 24000, 22050)
        audio_mask = interpolate_1d(y_mask, audio.shape[1])
        audio_lens = audio_mask.sum(-1)
        tokens = x
        token_lens = x_mask.sum(-1)

        self.aligner.eval()

        spec, spec_lens = self.aligner.preprocessor(input_signal=audio, length=rearrange(audio_lens, 'b 1 -> b'))
        _, attn_logprob = self.aligner(spec=spec, spec_len=spec_lens, text=tokens, text_len=token_lens) # (b, 1, ty', tx)
        attn_logprob = F.interpolate(attn_logprob, (spec_lens.max()*24000//22050, x.shape[1]), mode='nearest-exact')

        attn_soft = attn_logprob.clone()
        self.aligner.alignment_encoder._apply_mask(
            attn_soft,
            get_mask_from_lengths(token_lens).unsqueeze(-1) == 0,
            -float("inf")
        )
        attn_soft = self.aligner.alignment_encoder.softmax(attn_soft)

        attn_mas = binarize_attention(attn_soft, token_lens, spec_lens) # (b, 1, ty, tx)
        attn_hard = attn_mas.sum(-2)

        # rearrange to fit super() settings
        attn_hard = rearrange(attn_hard, 'b 1 tx -> b tx')
        attn_soft = rearrange(attn_soft, 'b 1 ty tx -> b tx ty')
        attn_mas = rearrange(attn_mas, 'b 1 ty tx -> b tx ty')
        return attn_hard, attn_soft, attn_logprob, attn_mas

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
        `cond` should be ground-truth duration instead of encoded audio
        """
        outputs = {}

        # text to phonemes, if tokenizer is given

        # TODO: deal with NeMo aligner's phoneme tokenizer
        assert exists(self_attn_mask) and exists(phoneme_ids) and exists(phoneme_len) and exists(phoneme_mask)

        phoneme_ids = phoneme_ids.clamp(min = 0)

        # get phoneme embeddings

        phoneme_emb = self.to_phoneme_emb(phoneme_ids)

        # aligner
        # use alignment_hard to oversample phonemes
        # Duration Predictor should predict the duration of unmasked phonemes where target is masked alignment_hard

        assert all([exists(el) for el in (phoneme_len, mel_len, phoneme_mask, mel_mask)]), 'need to pass phoneme_len, mel_len, phoneme_mask, mel_mask, to train duration predictor module'

        alignment_hard, alignment_soft, alignment_logprob, alignment_mas = self.forward_aligner(phoneme_ids, phoneme_mask, mel, mel_mask)
        target = alignment_hard
        outputs["target"] = target

        # create dummy cond when not given, become purely unconditional regression model

        if not exists(cond):
            if calculate_cond:
                cond = alignment_hard
            else:
                cond = torch.zeros_like(phoneme_ids)
                cond_drop_prob = 1

        cond = rearrange(cond, 'b t -> b t 1')
        cond = self.proj_in(cond)

        # construct mask if not given

        if not exists(cond_mask):
            batch, seq_len = phoneme_ids.shape
            cond_mask = self.create_cond_mask(batch=batch, seq_len=seq_len, training=self.training)

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
        outputs["durations"] = durations

        if not self.training:
            if return_aligned_phoneme_ids:
                aligned_phoneme_ids = self.align_phoneme_ids_with_durations(phoneme_ids, durations)
                outputs["aligned_phoneme_ids"] = aligned_phoneme_ids

            return outputs

        loss_mask = cond_mask & self_attn_mask

        loss = F.l1_loss(durations, target, reduction = 'none')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den
        loss = loss.mean()

        #aligner loss
        losses = {
            "dp": loss,
        }

        if return_aligned_phoneme_ids:
            aligned_phoneme_ids = self.align_phoneme_ids_with_durations(phoneme_ids=phoneme_ids, durations=target)
            outputs["aligned_phoneme_ids"] = aligned_phoneme_ids

        return loss, losses, outputs


class MFADurationPredictor(DurationPredictor):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        1. `cond` arg of `forward() should be ground-truth duration, therefore fix `self.proj_in` into nn.Linear(1,dim)
        """
        # audio_enc_dec = aligner.preprocessor

        del kwargs["aligner_kwargs"]
        super().__init__(
            *args,
            **kwargs,
            aligner_kwargs=None
        )

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
        self_attn_mask = None,
        return_aligned_phoneme_ids = False,
        **kwargs
    ):
        """
        Let ConditionalFlowMatcherWrapper handle:
            - cond
                - sample_rate matching
        """
        outputs = {}

        assert exists(target)
        outputs["target"] = target

        # text to phonemes, if tokenizer is given
        phoneme_ids = self.to_phoneme_ids(texts, phoneme_ids)

        assert exists(self_attn_mask) and exists(phoneme_len) and exists(phoneme_mask)

        if phoneme_mask.ndim == 2:
            phoneme_mask =  rearrange(self_attn_mask, 'b t -> b 1 t')

        # get phoneme embeddings
        phoneme_emb = self.to_phoneme_emb(phoneme_ids)

        assert exists(cond)

        cond = rearrange(cond, 'b t -> b t 1')
        cond = self.proj_in(cond)

        # construct mask if not given
        if not exists(cond_mask):
            batch, seq_len = phoneme_ids.shape
            cond_mask = self.create_cond_mask(batch=batch, seq_len=seq_len, training=self.training)
        outputs["cond_mask"] = cond_mask

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
        outputs["durations"] = durations

        if not self.training:
            if return_aligned_phoneme_ids:
                aligned_phoneme_ids = self.align_phoneme_ids_with_durations(phoneme_ids, durations)
                outputs["aligned_phoneme_ids"] = aligned_phoneme_ids

            return outputs

        loss_mask = cond_mask & self_attn_mask

        loss = F.l1_loss(durations, target, reduction = 'none')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den
        loss = loss.mean()

        losses = {
            "dp": loss,
        }

        def loss_masked(durations, target, *loss_masks):
            l_mask = reduce_masks_with_and(*loss_masks)
            loss_ = F.l1_loss(durations, target, reduction = 'none').masked_fill(~l_mask, 0.)
            loss_ = loss_.sum(-1) / l_mask.sum(-1).clamp(min=1e-5)
            loss_ = loss_.mean()
            return loss_

        # loss w/o sil, spn
        with torch.no_grad():
            if self.tokenizer.use_word_postfix:
                sil_mask = (phoneme_ids == 1) | (phoneme_ids == 2) | (phoneme_ids == 3) | (phoneme_ids == 4)
                spn_mask = (phoneme_ids == 5) | (phoneme_ids == 6) | (phoneme_ids == 7) | (phoneme_ids == 8)
            else:
                sil_mask = (phoneme_ids == 1)
                spn_mask = (phoneme_ids == 2)
            loss_sil = loss_masked(durations, target, loss_mask, sil_mask)
            loss_sil_spn = loss_masked(durations, target, loss_mask, sil_mask | spn_mask)
            loss_no_sil_spn = loss_masked(durations, target, loss_mask, ~sil_mask, ~spn_mask)

        losses.update({
            "dp_sil": loss_sil,
            "dp_sil_spn": loss_sil_spn,
            "dp_no_sil_spn": loss_no_sil_spn,
        })

        if return_aligned_phoneme_ids:
            aligned_phoneme_ids = self.align_phoneme_ids_with_durations(phoneme_ids=phoneme_ids, durations=target)
            outputs["aligned_phoneme_ids"] = aligned_phoneme_ids
        
        return loss, losses, outputs


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class VoiceBox(LightningModule):
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
        use_unet_skip_connection = False,
        skip_connect_scale = None,
        use_gateloop_layers = False,
        num_register_tokens = 16,
        p_drop_prob = 0.3, # p_drop in paper
        frac_lengths_mask: Tuple[float, float] = (0.7, 1.),
        condition_on_text = True,
        loss_masked = True,
        no_diffusion = False,
        # rmsnorm_shape = -1,
        fix_time_emb = False,
        text_encode = False,
        text_enc_depth = 4,
        text_enc_use_unet_skip_connection = False,
        text_enc_frame_concat = False,
        text_enc_vb_masked = True,
        **kwargs
    ):
        """
        Input related args:
            - audio_enc_dec: Optional[AudioEncoderDecoder] = None, for EnCodecVoco or MelVoco
            - dim_cond_emb = 1024,
            - dim = 1024,
                - dim_in = None, have to be None or equal to dim. should be deprecated and replaced with dim
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
            - p_drop_prob = 0.3, p_drop in paper
            - frac_lengths_mask: Tuple[float, float] = (0.7, 1.),
        """
        if text_encode:
            text_enc_depth = min(text_enc_depth, depth//2)
            depth = depth - text_enc_depth

        super().__init__()
        dim_in = default(dim_in, dim)

        time_hidden_dim = default(time_hidden_dim, dim * 4)

        self.audio_enc_dec = audio_enc_dec

        if exists(audio_enc_dec) and dim != audio_enc_dec.latent_dim:
            self.proj_in = nn.Linear(audio_enc_dec.latent_dim, dim)
        else:
            self.proj_in = nn.Identity()

        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU()
        )

        assert not (condition_on_text and not exists(num_cond_tokens)), 'number of conditioning tokens must be specified (whether phonemes or semantic token ids) if training conditional voicebox'

        if not condition_on_text:
            dim_cond_emb = 0

        self.dim_cond_emb = dim_cond_emb
        self.condition_on_text = condition_on_text
        self.num_cond_tokens = num_cond_tokens

        if condition_on_text:
            self.null_cond_id = num_cond_tokens # use last phoneme token as null token for CFG
            self.to_cond_emb = nn.Embedding(num_cond_tokens + 1, dim_cond_emb)

        self.p_drop_prob = p_drop_prob
        self.frac_lengths_mask = frac_lengths_mask

        self.to_embed = nn.Linear(dim_in * 2 + dim_cond_emb, dim)

        self.null_cond = nn.Parameter(torch.zeros(dim_in), requires_grad = False)

        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout,
            attn_dropout= attn_dropout,
            attn_flash = attn_flash,
            attn_qk_norm = attn_qk_norm,
            num_register_tokens = num_register_tokens,
            adaptive_rmsnorm = True,
            adaptive_rmsnorm_cond_dim_in = time_hidden_dim,
            use_unet_skip_connection = use_unet_skip_connection,
            skip_connect_scale = skip_connect_scale,
            use_gateloop_layers = use_gateloop_layers
        )

        dim_out = audio_enc_dec.latent_dim if exists(audio_enc_dec) else dim_in

        self.to_pred = nn.Linear(dim, dim_out, bias = False)

        ###
        self.audio_enc_dec.freeze()
        self.loss_masked = loss_masked

        self.no_diffusion = no_diffusion
        if self.no_diffusion:
            # A3T-like regression model
            dim_in = default(dim_in, dim)
            self.to_embed = nn.Linear(dim_in + self.dim_cond_emb, dim)

        self.fix_time_emb = fix_time_emb
        if self.fix_time_emb:
            self.sinu_pos_emb[0] = SinusoidalPosEmb(dim)
        
        self.code_project = isinstance(self.audio_enc_dec, DACVoco) and self.audio_enc_dec.return_code
        if self.code_project:
            assert self.no_diffusion
            # assert code_project_dim == self.audio_enc_dec.model.codebook_dim
            # codebook_size: 1024, codebook_dim: 8

            # mask_token: codebook_size
            self.mask_code_id = self.audio_enc_dec.model.codebook_size
            # what about pad_token?? -> zero

            # 8 -> proj_dim
            self.to_code_embs = nn.ModuleList([
                nn.Embedding(self.audio_enc_dec.model.codebook_size+1, self.audio_enc_dec.model.codebook_dim, padding_idx=self.mask_code_id, device=self.device)
                for _ in range(self.audio_enc_dec.model.n_codebooks)
            ])

            # cond_emb + cond_token_emb -> dim
            self.to_embed = nn.Linear(self.audio_enc_dec.latent_dim + self.dim_cond_emb, dim)

            # self.to_pred: proj_dim -> 96
            # to_code: 96 -> 12 cls heads
            self.to_code = nn.Linear(self.audio_enc_dec.latent_dim, self.audio_enc_dec.model.n_codebooks * self.audio_enc_dec.model.codebook_size, device=self.device)

        self.text_encode = text_encode
        if self.text_encode:
            self.text_encoder = Transformer(
                dim = dim,
                depth = text_enc_depth,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                ff_dropout = ff_dropout,
                attn_dropout= attn_dropout,
                attn_flash = attn_flash,
                attn_qk_norm = attn_qk_norm,
                num_register_tokens = num_register_tokens,
                adaptive_rmsnorm = False,
                use_unet_skip_connection = text_enc_use_unet_skip_connection,
                skip_connect_scale = skip_connect_scale,
                use_gateloop_layers = use_gateloop_layers
            )
            self.text_enc_frame_concat = text_enc_frame_concat
            if self.text_enc_frame_concat:
                # audio: self.proj_in -> dim
                # text: self.to_cond_emb -> dim_cond_emb
                self.text_audio_to_embed = nn.Linear(dim + dim_cond_emb, dim)
                self.text_audio_conv_embed = ConvPositionEmbed(
                    dim = dim,
                    kernel_size = conv_pos_embed_kernel_size,
                    groups = conv_pos_embed_groups,
                )
            else:
                self.text_to_embed = nn.Linear(dim_cond_emb, dim)
                self.text_audio_conv_embed = ConvPositionEmbed(
                    dim = dim,
                    kernel_size = conv_pos_embed_kernel_size,
                    groups = conv_pos_embed_groups,
                )
            self.text_enc_vb_masked = text_enc_vb_masked
            self.proj_out = nn.Linear(dim, audio_enc_dec.latent_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def create_cond_mask(self, batch, seq_len, cond_token_ids=None, self_attn_mask=None, training=True, frac_lengths_mask=None, phn_bnd_eps=None):
        if training:
            frac_lengths_mask = default(frac_lengths_mask, self.frac_lengths_mask)
            frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*frac_lengths_mask)
            cond_mask = self.phone_level_mask_from_frac_lengths(seq_len, frac_lengths, cond_token_ids, self_attn_mask, phn_bnd_eps)
        else:
            cond_mask = torch.zeros((batch, seq_len), device = self.device, dtype = torch.bool)
        return cond_mask

    @torch.no_grad()
    def phone_level_mask_from_frac_lengths(
        self,
        seq_len: int,
        frac_lengths: Tensor,
        cond_token_ids: Tensor,
        self_attn_mask: None | Tensor = None,
        phn_bnd_eps: None | int | float = None,
    ):
        device = frac_lengths.device

        if exists(self_attn_mask):
            _seq_len = seq_len
            seq_len = self_attn_mask.sum(-1)

        lengths = (frac_lengths * seq_len).long()
        max_start = seq_len - lengths

        rand = torch.zeros_like(frac_lengths, device = device).float().uniform_(0, 1)
        start = (max_start * rand).clamp(min = 0)
        end = start + lengths

        if phn_bnd_eps is None:
            if exists(self_attn_mask):
                end = torch.minimum(end, seq_len)
                seq_len = _seq_len
            else:
                end = end.clamp(max = seq_len)
            return mask_from_start_end_indices(seq_len, start, end)

        # start = torch.floor(start).long()
        start_of_start = self.find_start_of_phone(cond_token_ids, torch.floor(start).long())
        end_of_start = self.find_end_of_phone(cond_token_ids, torch.floor(start).long(), seq_len)
        prob = (start - start_of_start) / (end_of_start - start_of_start + 1)
        _start = torch.where(prob > 0.5, end_of_start + 1, start_of_start)

        # end = torch.floor(end).long()
        if exists(self_attn_mask):
            end = torch.minimum(end, seq_len-1)
        start_of_end = self.find_start_of_phone(cond_token_ids, torch.floor(end).long())
        end_of_end = self.find_end_of_phone(cond_token_ids, torch.floor(end).long(), seq_len)
        prob = (end - start_of_end) / (end_of_end - start_of_end + 1)
        _end = torch.where(prob > 0.5, end_of_end + 1, start_of_end)

        start = (torch.minimum(_start, _end) - phn_bnd_eps).clamp(min = 0)
        end = torch.maximum(_start, _end) + phn_bnd_eps

        if exists(self_attn_mask):
            end = torch.minimum(end, seq_len)
            seq_len = _seq_len
        else:
            end = end.clamp(max = seq_len)

        return mask_from_start_end_indices(seq_len, start, end)

    @torch.no_grad()
    def find_start_of_phone(self, cond_token_ids, idx):
        phone_token_id = cond_token_ids[range(len(idx)), idx]
        start_of_phone = idx
        should_continue = (start_of_phone > 0)
        while should_continue.sum() > 0:
            prev_idx = torch.where(should_continue, start_of_phone - 1, start_of_phone)
            prev_the_same = should_continue & (cond_token_ids[range(len(idx)), prev_idx] == phone_token_id)
            start_of_phone = torch.where(prev_the_same, start_of_phone - 1, start_of_phone)
            should_continue = prev_the_same & (start_of_phone > 0)
        return start_of_phone

    @torch.no_grad()
    def find_end_of_phone(self, cond_token_ids, idx, seq_len):
        phone_token_id = cond_token_ids[range(len(idx)), idx]
        end_of_phone = idx
        should_continue = (end_of_phone < seq_len - 1)
        while should_continue.sum() > 0:
            post_idx = torch.where(should_continue, end_of_phone + 1, end_of_phone)
            post_the_same = should_continue & (cond_token_ids[range(len(idx)), post_idx] == phone_token_id)
            end_of_phone = torch.where(post_the_same, end_of_phone + 1, end_of_phone)
            should_continue = post_the_same & (end_of_phone < seq_len - 1)
        return end_of_phone

    def cond_project_and_mask(self, cond, cond_mask_with_pad_dim, self_attn_mask):
        # discrete code input + regression
        if self.code_project:
            # inputs: x=None, cond = discrete code, (b,t,n)
            # output: code classification

            # mask -> mask token
            cond = torch.where(
                cond_mask_with_pad_dim,
                self.mask_code_id,
                cond
            )

            # to emb
            conds = [
                self.to_code_embs[i](cond[:, :, i]) * (i < self.audio_enc_dec.bandwidth_id)
                for i in range(self.audio_enc_dec.model.n_codebooks)
            ]
            cond = torch.concat(conds, dim=-1)

            # remove padding
            cond = cond * rearrange(self_attn_mask, 'b t -> b t 1')

        elif exists(cond):
            cond = self.proj_in(cond)

            # as described in section 3.2
            cond = cond * ~cond_mask_with_pad_dim
        
        return cond

    def forward(
        self,
        x,
        *,
        times,
        cond_token_ids,
        self_attn_mask = None,
        cond_drop_prob = 0.1,
        target = None,
        cond = None,
        cond_mask: BoolTensor | None = None
    ):
        """ Copied from `super.forward()`.

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
        outputs = {}

        if self.no_diffusion:
            # x = None, target = cond if training else None
            pass

        else:
            # project in, in case codebook dim is not equal to model dimensions
            x = self.proj_in(x)

        cond = default(cond, target)
        outputs["cond"] = cond

        # shapes

        batch, seq_len, _ = cond.shape

        # construct conditioning mask if not given

        if not exists(cond_mask):
            cond_mask = self.create_cond_mask(batch=batch, seq_len=seq_len, cond_token_ids=cond_token_ids, self_attn_mask=self_attn_mask, training=self.training)

        cond_mask_with_pad_dim = rearrange(cond_mask, '... -> ... 1')
        outputs["cond_mask"] = cond_mask_with_pad_dim

        # as described in section 3.2
        cond = self.cond_project_and_mask(cond, cond_mask_with_pad_dim, self_attn_mask)

        # auto manage shape of times, for odeint times

        if times.ndim == 0:
            times = repeat(times, '-> b', b = cond.shape[0])

        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = cond.shape[0])

        if self.no_diffusion:
            times = torch.zeros_like(times)

        cond_ids = cond_token_ids

        if not self.no_diffusion:

            # classifier free guidance

            if cond_drop_prob > 0.:
                cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, self.device)

                cond = torch.where(
                    rearrange(cond_drop_mask, '... -> ... 1 1'),
                    self.null_cond,
                    cond
                )

                cond_ids = torch.where(
                    rearrange(cond_drop_mask, '... -> ... 1'),
                    self.null_cond_id,
                    cond_token_ids
                )

            # spectrogram dropout
            
            if self.training:
                p_drop_mask = prob_mask_like(cond.shape[:1], self.p_drop_prob, self.device)

                cond = torch.where(
                    rearrange(p_drop_mask, '... -> ... 1 1'),
                    self.null_cond,
                    cond
                )
                
                p_drop_mask = prob_mask_like(cond.shape[:1], self.p_drop_prob, self.device)
                times = times * ~p_drop_mask

        # phoneme or semantic conditioning embedding

        cond_emb = None

        if self.condition_on_text:
            cond_emb = self.to_cond_emb(cond_ids)

            cond_emb_length = cond_emb.shape[-2]
            if cond_emb_length != seq_len:
                cond_emb = rearrange(cond_emb, 'b n d -> b d n')
                cond_emb = interpolate_1d(cond_emb, seq_len)
                cond_emb = rearrange(cond_emb, 'b d n -> b n d')

                if exists(self_attn_mask):
                    self_attn_mask = interpolate_1d(self_attn_mask, seq_len)

        # concat source signal, semantic / phoneme conditioning embed, and conditioning
        # and project

        if self.text_encode:
            if self.text_enc_frame_concat:
                to_concat = [*filter(exists, (cond, cond_emb))]
                conv_cond_text = torch.cat(to_concat, dim = 2)
                conv_cond_text = self.text_audio_to_embed(conv_cond_text)
                conv_cond_text = self.text_audio_conv_embed(conv_cond_text, mask=~self_attn_mask)
                cond = self.text_encoder(conv_cond_text, mask=self_attn_mask)
            else:
                cond_text = self.text_to_embed(cond_emb)
                conv_cond = self.text_audio_conv_embed(cond, mask=~self_attn_mask)
                conv_text = self.text_audio_conv_embed(cond_text, mask=~self_attn_mask)
                to_concat = [*filter(exists, (conv_cond, conv_text))]
                conv_cond_text = torch.cat(to_concat, dim = 1)
                mask = torch.cat([~cond_mask * self_attn_mask, self_attn_mask], dim=1)
                cond = self.text_encoder(conv_cond_text, mask=mask)
                cond = cond[:, seq_len:]
            pred_ori_cond = self.proj_out(cond)
            outputs["pred_ori_cond"] = pred_ori_cond

        to_concat = [*filter(exists, (x, cond_emb, cond))]
        embed = torch.cat(to_concat, dim = -1)

        x = self.to_embed(embed)

        x = self.conv_embed(x, mask=~self_attn_mask) + x

        time_emb = self.sinu_pos_emb(times)

        # attend

        if self.text_encode and self.text_enc_vb_masked:
            x = self.transformer(
                x,
                mask = cond_mask,
                adaptive_rmsnorm_cond = time_emb
            )
        else:
            x = self.transformer(
                x,
                mask = self_attn_mask,
                adaptive_rmsnorm_cond = time_emb
            )

        x = self.to_pred(x)

        if self.code_project:
            x = self.to_code(x).reshape(batch, seq_len, self.audio_enc_dec.model.n_codebooks, self.audio_enc_dec.model.codebook_size)
            outputs["pred"] = torch.argmax(x, dim=-1)
            x = rearrange(x, 'b t n c -> b c t n')
            # TODO: classification loss
        else:
            outputs["pred"] = x

        # if no target passed in, just return logits

        if not exists(target):
            return outputs["pred"]

        if self.loss_masked:
            loss_mask = reduce_masks_with_and(cond_mask, self_attn_mask)
        else:
            loss_mask = self_attn_mask

        if not exists(loss_mask):
            if self.code_project:
                # x: (b,c,t,n), target: (b,t,n)
                return F.cross_entropy(x[:, :, :, :self.audio_enc_dec.bandwidth_id], target[:, :, :self.audio_enc_dec.bandwidth_id]), outputs
            elif isinstance(self.audio_enc_dec, DACVoco):
                # (b,t,d)
                return F.mse_loss(x[:, :, :self.audio_enc_dec.masked_latent_dim], target[:, :, :self.audio_enc_dec.masked_latent_dim]), outputs
            else:
                return F.mse_loss(x, target), outputs

        if self.code_project:
            # x: (b,c,t,n), target: (b,t,n)
            loss = F.cross_entropy(x[:, :, :, :self.audio_enc_dec.bandwidth_id], target[:, :, :self.audio_enc_dec.bandwidth_id], reduction = 'none')
        elif isinstance(self.audio_enc_dec, DACVoco):
            # (b,t,d)
            loss = F.mse_loss(x[:, :, :self.audio_enc_dec.masked_latent_dim], target[:, :, :self.audio_enc_dec.masked_latent_dim], reduction = 'none')
        else:
            loss = F.mse_loss(x, target, reduction = 'none')

        # TODO: weighted loss for different codes? or simply use less codes?
        loss = reduce(loss, 'b n d -> b n', 'mean')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den

        outputs["loss_mask"] = loss_mask
        outputs["self_attn_mask"] = self_attn_mask

        if self.text_encode:
            assert self.loss_masked
            text_enc_loss = F.mse_loss(pred_ori_cond, outputs["cond"], reduction='none')
            text_enc_loss = reduce(text_enc_loss, 'b n d -> b n', 'mean')
            text_enc_loss = text_enc_loss.masked_fill(~loss_mask, 0.)
            num = reduce(text_enc_loss, 'b n -> b', 'sum')
            den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
            text_enc_loss = num / den
            outputs["text_enc_loss"] = text_enc_loss.mean()

        return loss.mean(), outputs
    

class ConditionalFlowMatcherWrapper(LightningModule):
    """ Deal with `self.forward()` duration prediction and aligner.
    """
    @beartype
    def __init__(
        self,
        voicebox: VoiceBox,
        text_to_semantic: None = None,
        duration_predictor: Optional[DurationPredictor] = None,
        sigma = 0.,
        ode_atol = 1e-5,
        ode_rtol = 1e-5,
        use_torchode = False,
        torchdiffeq_ode_method = 'midpoint',   # use midpoint for torchdiffeq, as in paper
        torchode_method_klass = to.Tsit5,      # use tsit5 for torchode, as torchode does not have midpoint (recommended by Bryan @b-chiang)
        cond_drop_prob = 0.,
        **kwargs
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

            - use_torchode = False,

            - torchdiffeq_ode_method = 'midpoint',

                use midpoint for torchdiffeq, as in paper

            - torchode_method_klass = to.Tsit5,

                use tsit5 for torchode, as torchode does not have midpoint (recommended by Bryan @b-chiang)

        Voicebox forward args

            - sigma = 0.,

            - cond_drop_prob = 0.
        """
        super().__init__()
        self.sigma = sigma

        self.voicebox = voicebox
        self.condition_on_text = voicebox.condition_on_text

        assert not (not self.condition_on_text and exists(text_to_semantic)), 'TextToSemantic should not be passed in if not conditioning on text'
        assert not (exists(text_to_semantic) and not exists(text_to_semantic.wav2vec)), 'the wav2vec module must exist on the TextToSemantic, if being used to condition on text'

        self.text_to_semantic = text_to_semantic
        self.duration_predictor = duration_predictor

        if self.condition_on_text:
            assert exists(text_to_semantic) ^ exists(duration_predictor), 'you should use either TextToSemantic from Spear-TTS, or DurationPredictor for the text / phoneme to audio alignment, but not both'

        self.cond_drop_prob = cond_drop_prob

        self.use_torchode = use_torchode
        self.torchode_method_klass = torchode_method_klass

        self.odeint_kwargs = dict(
            atol = ode_atol,
            rtol = ode_rtol,
            method = torchdiffeq_ode_method
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path, strict = True):
        # return pkg so the trainer can access it
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(pkg['model'], strict = strict)
        return pkg

    @torch.inference_mode()
    def sample(
        self,
        *_args,
        dp_cond = None,
        cond = None,
        self_attn_mask = None,
        texts: Optional[List[str]] = None,
        text_token_ids: Optional[Tensor] = None,
        semantic_token_ids: Optional[Tensor] = None,
        phoneme_ids: Optional[Tensor] = None,
        aligned_phoneme_ids: Optional[Tensor] = None,
        cond_mask = None,
        steps = 3,
        cond_scale = 1.,
        decode_to_audio = True,
        max_semantic_token_ids = 2048,
        spec_decode = False,
        spec_decode_gamma = 5, # could be higher, since speech is probably easier than text, needs to be tested
        sample_std = 1.,
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
            
            # assert audio sampling_rate == voicebox sampling_rate
            cond = self.voicebox.audio_enc_dec.encode(cond)

        # setup text conditioning, either coming from duration model (as phoneme ids)
        # for coming from text-to-semantic module from spear-tts paper, as (semantic ids)

        num_cond_inputs = sum([*map(exists, (texts, text_token_ids, semantic_token_ids, phoneme_ids, aligned_phoneme_ids))])
        assert num_cond_inputs <= 1

        cond_token_ids = None

        if self.condition_on_text:
            assert not (exists(self.text_to_semantic) or exists(semantic_token_ids))
            assert exists(self.duration_predictor)

            if not exists(aligned_phoneme_ids):
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

                # TODO: why not interpolate???
                cond = curtail_or_pad(cond, cond_target_length)
                # self_attn_mask = curtail_or_pad(torch.ones_like(cond, dtype=torch.bool), cond_target_length)
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

            # print(x.shape, t.shape, cond_token_ids.shape, cond.shape, cond_mask.shape)
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

        y0 = torch.randn_like(cond) * sample_std
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

    @torch.no_grad()
    def parse_vb_input(self, x1, mask, cond, input_sampling_rate=None):
        audio_enc_dec_sampling_rate = self.voicebox.audio_enc_dec.sampling_rate
        input_sampling_rate = default(input_sampling_rate, audio_enc_dec_sampling_rate)
        input_is_raw_audio, cond_is_raw_audio = map(is_probably_audio_from_shape, (x1, cond))
        if any([input_is_raw_audio, cond_is_raw_audio]):
            assert exists(self.voicebox.audio_enc_dec), 'audio_enc_dec must be set on VoiceBox to train directly on raw audio'
            self.voicebox.audio_enc_dec.eval()
            audio_enc_dec_sampling_rate = self.voicebox.audio_enc_dec.sampling_rate
        
            if input_is_raw_audio:
                x1 = resample(x1, input_sampling_rate, audio_enc_dec_sampling_rate)
                x1 = self.voicebox.audio_enc_dec.encode(x1)

            if exists(cond) and cond_is_raw_audio:
                cond = resample(cond, input_sampling_rate, audio_enc_dec_sampling_rate)
                cond = self.voicebox.audio_enc_dec.encode(cond)

            audio_len = mask.sum(-1)
            # mel_len = audio_len * x1.shape[1] // mask.shape[-1]
            mel_len = torch.ceil(audio_len / self.voicebox.audio_enc_dec.downsample_factor)
            mask = get_mask_from_lengths(mel_len)

        return {
            'x1': x1,
            'cond': cond,
            'mask': mask
        }
    
    def forward(
        self,
        x1,
        *,
        mask = None,
        phoneme_ids = None,
        cond = None,
        cond_mask = None,
        input_sampling_rate = None # will assume it to be the same as the audio encoder decoder sampling rate, if not given. if given, will resample
    ):
        """TODO: Deal with phoneme duration alignment and expansion

        Args:
            - x1, input audio    
            - mask = None, audio_mask (self_attn_mask)
            - semantic_token_ids = None, pass in if using semantic tokens as text input
            - phoneme_ids = None, pass in if using phoneme sequence as text input
            - cond = None, for context audio, could set to x1 or None
            - cond_mask = None, masking context audio. Could pass None and let Voicebox generate for you
            - input_sampling_rate = None, resample if given & != vocoder sampling rate
        """

        batch, seq_len, dtype, σ = *x1.shape[:2], x1.dtype, self.sigma

        # setup text conditioning, either coming from duration model (as phoneme ids)
        # or from text-to-semantic module, semantic ids encoded with wav2vec (hubert usually)

        assert self.condition_on_text or not exists(phoneme_ids), 'phoneme ids should not be passed in if not conditioning on text'


        self_attn_mask = mask
        cond_token_ids = None

        # handle downsample audio_mask

        if self.condition_on_text and exists(self.duration_predictor):
            assert not exists(self.text_to_semantic)
            assert exists(phoneme_ids)
            cond_token_ids = phoneme_ids

        else:
            assert not exists(phoneme_ids), 'no conditioning inputs should be given if not conditioning on text'

        # regression

        if self.voicebox.no_diffusion:

            # zero times

            times = torch.zeros((batch,), dtype = dtype, device = self.device)

            # predict

            if not self.voicebox.training:
                vb_pred = self.voicebox(
                    x=None,
                    cond = cond,
                    cond_mask = cond_mask,
                    times = times,
                    target = None,
                    self_attn_mask = self_attn_mask,
                    cond_token_ids = cond_token_ids,
                    cond_drop_prob = self.cond_drop_prob
                )
                return vb_pred

            else:
                loss, vb_outputs = self.voicebox(
                    x=None,
                    cond = cond,
                    cond_mask = cond_mask,
                    times = times,
                    target = cond,
                    self_attn_mask = self_attn_mask,
                    cond_token_ids = cond_token_ids,
                    cond_drop_prob = self.cond_drop_prob
                )

        else:
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

            # self.voicebox.train()

            loss, vb_outputs = self.voicebox(
                w,
                cond = cond,
                cond_mask = cond_mask,
                times = times,
                target = flow,
                self_attn_mask = self_attn_mask,
                cond_token_ids = cond_token_ids,
                cond_drop_prob = self.cond_drop_prob
            )
            vb_outputs.update({
                'x1': x1,
                'x0': x0,
                'w': w,
                'flow': flow
            })

        losses = {}
        losses['vb'] = loss

        outputs = {
            "vb": vb_outputs,
        }
        if "text_enc_loss" in vb_outputs:
            losses['text_enc'] = vb_outputs["text_enc_loss"]
            del vb_outputs["text_enc_loss"]

        return loss, losses, outputs

    def waveform_loss(self, outputs, audio, audio_mask):
        if self.voicebox.no_diffusion:
            pred_x1 = outputs['vb']['pred']
        else:
            x0, pred_dx = outputs['vb']['x0'], outputs['vb']['pred']
            σ = self.sigma
            pred_x1 = pred_dx + (1 - σ) * x0
        with torch.set_grad_enabled(True):
            pred_audio = self.voicebox.audio_enc_dec.decode(pred_x1)

        loss_mask = outputs['vb']['loss_mask']
        # cond, cond_mask = outputs['vb']["cond"], outputs['vb']["cond_mask"]
        #TODO: feature mask -> waveform mask

        # mel_len
        hop_size = self.voicebox.audio_enc_dec.downsample_factor
        audio_loss_mask = self.duration_predictor.align_phoneme_ids_with_durations(loss_mask, torch.ones_like(loss_mask)*hop_size).bool()
        max_audio_len = min(audio_mask.shape[-1], audio_loss_mask.shape[-1])
        audio_loss_mask = audio_loss_mask[:, :max_audio_len] & audio_mask[:, :max_audio_len]

        loss = F.l1_loss(pred_audio[:, :max_audio_len], audio[:, :max_audio_len], reduction='none')
        # loss = reduce(loss, 'b n d -> b n', 'mean')
        loss = loss.masked_fill(~audio_loss_mask, 0.)

        # masked mean

        num = reduce(loss, 'b n -> b', 'sum')
        den = audio_loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den
        return loss.mean()

    def cross_entropy_loss(self, outputs, audio, audio_mask):
        if self.voicebox.no_diffusion:
            pred_x1 = outputs['vb']['pred']
        else:
            x0, pred_dx = outputs['vb']['x0'], outputs['vb']['pred']
            σ = self.sigma
            pred_x1 = pred_dx + (1 - σ) * x0
        x1 = outputs['vb']['x1']

        loss = self.voicebox.audio_enc_dec.cross_entropy_loss(pred_x1, x1)

        loss_mask = outputs['vb']['loss_mask']

        # masked mean
        # (b, t, n_codebooks)
        loss = reduce(loss, 'b t d -> b t', 'mean')
        loss = loss.masked_fill(~loss_mask, 0.)

        num = reduce(loss, 'b t -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den

        return loss.mean()