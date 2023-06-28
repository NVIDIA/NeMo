# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import contextlib
import itertools
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger

from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.fastpitchloss import DurationLoss, EnergyLoss, MelLoss, PitchLoss
from nemo.collections.tts.losses.hifigan_losses import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.modules.fastpitch import FastPitchModule
from nemo.collections.tts.modules.hifigan_modules import MultiPeriodDiscriminator, MultiScaleDiscriminator
from nemo.collections.tts.parts.utils.helpers import (
    clip_grad_value_,
    get_batch_size,
    get_num_workers,
    plot_spectrogram_to_numpy,
    process_batch,
    rand_slice_segments,
    slice_segments,
)
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    Index,
    LengthsType,
    MelSpectrogramType,
    ProbsType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.lr_scheduler import CosineAnnealing, compute_max_steps
from nemo.utils import logging, model_utils
from nemo.utils.decorators.experimental import experimental

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


@dataclass
class G2PConfig:
    _target_: str = "nemo_text_processing.g2p.modules.EnglishG2p"
    phoneme_dict: str = "scripts/tts_dataset_files/cmudict-0.7b_nv22.10"
    heteronyms: str = "scripts/tts_dataset_files/heteronyms-052722"
    phoneme_probability: float = 0.5


@dataclass
class TextTokenizer:
    _target_: str = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.EnglishPhonemesTokenizer"
    punct: bool = True
    stresses: bool = True
    chars: bool = True
    apostrophe: bool = True
    pad_with_space: bool = True
    add_blank_at: bool = True
    g2p: G2PConfig = G2PConfig()

@dataclass
class VocoderModel:
    generator = None
    msd = None
    mpd = None


@dataclass
class TextTokenizerConfig:
    text_tokenizer: TextTokenizer = TextTokenizer()

@experimental
class FastPitchE2EModel(TextToWaveform):
    """
    FastPitch model (https://arxiv.org/abs/2006.06873) that is used to generate mel spectrogram from text 
    with HiFi-GAN model (https://arxiv.org/abs/2010.05646) to generate audios.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Setup normalizer
        self.normalizer = None
        self.text_normalizer_call = None
        self.text_normalizer_call_kwargs = {}
        self._setup_normalizer(cfg)

        self.learn_alignment = cfg.get("learn_alignment", False)

        # Setup vocabulary (=tokenizer) and input_fft_kwargs (supported only with self.learn_alignment=True)
        input_fft_kwargs = {}
        if self.learn_alignment:
            self.tokenizer = None
            self.ds_class_name = cfg.train_ds.dataset._target_.split(".")[-1]

            self._setup_tokenizer(cfg)
            assert self.tokenizer is not None
            input_fft_kwargs["n_embed"] = len(self.tokenizer.tokens)
            input_fft_kwargs["padding_idx"] = self.tokenizer.pad

        self._parser = None
        self._tb_logger = None
        super().__init__(cfg=cfg, trainer=trainer)

        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 100)
        self.log_train_images = False

        loss_scale = 0.1 if self.learn_alignment else 1.0
        dur_loss_scale = loss_scale
        pitch_loss_scale = loss_scale
        energy_loss_scale = loss_scale
        if "dur_loss_scale" in cfg:
            dur_loss_scale = cfg.dur_loss_scale
        if "pitch_loss_scale" in cfg:
            pitch_loss_scale = cfg.pitch_loss_scale
        if "energy_loss_scale" in cfg:
            energy_loss_scale = cfg.energy_loss_scale

        self.mel_loss_fn = MelLoss()
        self.pitch_loss_fn = PitchLoss(loss_scale=pitch_loss_scale)
        self.duration_loss_fn = DurationLoss(loss_scale=dur_loss_scale)
        self.energy_loss_fn = EnergyLoss(loss_scale=energy_loss_scale)

        self.feature_loss = FeatureMatchingLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

        self.aligner = None
        if self.learn_alignment:
            self.aligner = instantiate(self._cfg.alignment_module)
            self.forward_sum_loss_fn = ForwardSumLoss()
            self.bin_loss_fn = BinLoss()

        self.l1_factor = cfg.get("l1_loss_factor", 45)

        self.sample_rate = self._cfg.preprocessor.sample_rate
        self.stft_bias = None

        self.preprocessor = instantiate(self._cfg.preprocessor)


        if cfg.use_pretrain:
            if cfg.fastpitch_model is None or cfg.hifigan_model is None:
                raise ValueError(f'Should provide model paths if using pretrain')
            
            self.register_nemo_submodule(
                "hfg_model",
                config_field="hfg_model",
                model=HifiGanModel.from_pretrained(cfg.fastpitch_model, map_location=torch.device("cpu")),
            )

            self.register_nemo_submodule(
                "fp_model",
                config_field="fp_model",
                model=FastPitchModel.from_pretrained(cfg.hifigan_mode, map_location=torch.device("cpu")),
            )

            self.tokenizer = self.fp_model.vocab
            self.preprocessor = self.fp_model.preprocessor
        else:
            if (
                not "input_fft" in cfg
                or not "output_fft" in cfg
                or not "duration_predictor" in cfg
                or not "pitch_predictor" in cfg
                or not "generator" in cfg
            ):
                raise ValueError(f'Should provide model config if not using pretrain')

            input_fft = instantiate(self._cfg.input_fft, **input_fft_kwargs)
            output_fft = instantiate(self._cfg.output_fft)
            duration_predictor = instantiate(self._cfg.duration_predictor)
            pitch_predictor = instantiate(self._cfg.pitch_predictor)
            speaker_encoder = instantiate(self._cfg.get("speaker_encoder", None))
            speaker_emb_condition_prosody = cfg.get("speaker_emb_condition_prosody", False)
            speaker_emb_condition_decoder = cfg.get("speaker_emb_condition_decoder", False)
            speaker_emb_condition_aligner = cfg.get("speaker_emb_condition_aligner", False)
            energy_embedding_kernel_size = cfg.get("energy_embedding_kernel_size", 0)
            energy_predictor = instantiate(self._cfg.get("energy_predictor", None))
            
            n_speakers = cfg.get("n_speakers", 0)
            speaker_emb_condition_prosody = cfg.get("speaker_emb_condition_prosody", False)
            speaker_emb_condition_decoder = cfg.get("speaker_emb_condition_decoder", False)
            speaker_emb_condition_aligner = cfg.get("speaker_emb_condition_aligner", False)
            min_token_duration = cfg.get("min_token_duration", 0)
            use_log_energy = cfg.get("use_log_energy", True)
            if n_speakers > 1 and "add" not in input_fft.cond_input.condition_types:
                input_fft.cond_input.condition_types.append("add")
            if speaker_emb_condition_prosody:
                duration_predictor.cond_input.condition_types.append("add")
                pitch_predictor.cond_input.condition_types.append("add")
            if speaker_emb_condition_decoder:
                output_fft.cond_input.condition_types.append("add")
            if speaker_emb_condition_aligner and self.aligner is not None:
                self.aligner.cond_input.condition_types.append("add")

            self.fp_model = FastPitchModule(
                input_fft,
                output_fft,
                duration_predictor,
                pitch_predictor,
                energy_predictor,
                self.aligner,
                speaker_encoder,
                n_speakers,
                cfg.symbols_embedding_dim,
                cfg.pitch_embedding_kernel_size,
                energy_embedding_kernel_size,
                cfg.n_mel_channels,
                min_token_duration,
                cfg.max_token_duration,
                use_log_energy,
            )
            
            # [TODO] PTL fails to move it ro cerrect device himself, don't know if there is good way to do it
            generator = instantiate(cfg.generator).cuda()
            mpd = MultiPeriodDiscriminator(debug=False).cuda()
            msd = MultiScaleDiscriminator(debug=False).cuda()

            self.hfg_model = VocoderModel
            self.hfg_model.generator = generator
            self.hfg_model.mpd = mpd
            self.hfg_model.msd = msd

        self._input_types = self._output_types = None

        self.automatic_optimization = False

    def _get_max_steps(self):
        return compute_max_steps(
            max_epochs=self._cfg.max_epochs,
            accumulate_grad_batches=self.trainer.accumulate_grad_batches,
            limit_train_batches=self.trainer.limit_train_batches,
            num_workers=get_num_workers(self.trainer),
            num_samples=len(self._train_dl.dataset),
            batch_size=get_batch_size(self._train_dl),
            drop_last=self._train_dl.drop_last,
        )

    @staticmethod
    def get_warmup_steps(max_steps, warmup_steps, warmup_ratio):
        if warmup_steps is not None and warmup_ratio is not None:
            raise ValueError(f'Either use warmup_steps or warmup_ratio for scheduler')

        if warmup_steps is not None:
            return warmup_steps

        if warmup_ratio is not None:
            return warmup_ratio * max_steps

        raise ValueError(f'Specify warmup_steps or warmup_ratio for scheduler')

    def configure_optimizers(self):
        optim_config = self._cfg.optim.copy()

        OmegaConf.set_struct(optim_config, False)
        sched_config = optim_config.pop("sched", None)
        OmegaConf.set_struct(optim_config, True)

        optim_g = instantiate(
            optim_config, params=itertools.chain(self.hfg_model.generator.parameters(), self.fp_model.parameters()),
        )
        optim_d = instantiate(
            optim_config, params=itertools.chain(self.hfg_model.msd.parameters(), self.hfg_model.mpd.parameters()),
        )

        # Backward compatibility
        if sched_config is None and 'sched' in self._cfg:
            sched_config = self._cfg.sched

        if sched_config is not None:
            max_steps = self._cfg.get("max_steps", None)
            if max_steps is None or max_steps < 0:
                max_steps = self._get_max_steps()

            warmup_steps = FastPitchE2EModel.get_warmup_steps(
                max_steps=max_steps,
                warmup_steps=sched_config.get("warmup_steps", None),
                warmup_ratio=sched_config.get("warmup_ratio", None),
            )

            scheduler_g = CosineAnnealing(
                optimizer=optim_g, max_steps=max_steps, min_lr=sched_config.min_lr, warmup_steps=warmup_steps,
            )  # Use warmup to delay start
            sch1_dict = {
                'scheduler': scheduler_g,
                'interval': 'step',
            }

            scheduler_d = CosineAnnealing(optimizer=optim_d, max_steps=max_steps, min_lr=sched_config.min_lr,)
            sch2_dict = {
                'scheduler': scheduler_d,
                'interval': 'step',
            }

            return [optim_g, optim_d], [sch1_dict, sch2_dict]
        else:
            return [optim_g, optim_d]

    def _get_default_text_tokenizer_conf(self):
        text_tokenizer: TextTokenizerConfig = TextTokenizerConfig()
        return OmegaConf.create(OmegaConf.to_yaml(text_tokenizer))

    def _setup_normalizer(self, cfg):
        if "text_normalizer" in cfg:
            normalizer_kwargs = {}

            if "whitelist" in cfg.text_normalizer:
                normalizer_kwargs["whitelist"] = self.register_artifact(
                    'text_normalizer.whitelist', cfg.text_normalizer.whitelist
                )
            try:
                import nemo_text_processing

                self.normalizer = instantiate(cfg.text_normalizer, **normalizer_kwargs)
            except Exception as e:
                logging.error(e)
                raise ImportError(
                    "`nemo_text_processing` not installed, see https://github.com/NVIDIA/NeMo-text-processing for more details"
                )

            self.text_normalizer_call = self.normalizer.normalize
            if "text_normalizer_call_kwargs" in cfg:
                self.text_normalizer_call_kwargs = cfg.text_normalizer_call_kwargs

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

        self.tokenizer = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)

    @property
    def parser(self):
        if self._parser is not None:
            return self._parser

        if self.learn_alignment:
            ds_class_name = self._cfg.train_ds.dataset._target_.split(".")[-1]

            if ds_class_name == "TTSDataset":
                self._parser = self.tokenizer.encode
            elif ds_class_name == "AudioToCharWithPriorAndPitchDataset":
                if self.tokenizer is None:
                    tokenizer_conf = self._get_default_text_tokenizer_conf()
                    self._setup_tokenizer(tokenizer_conf)
                self._parser = self.tokenizer.encode
            else:
                raise ValueError(f"Unknown dataset class: {ds_class_name}")
        else:
            self._parser = parsers.make_parser(
                labels=self._cfg.labels,
                name='en',
                unk_id=-1,
                blank_id=-1,
                do_normalize=True,
                abbreviation_version="fastpitch",
                make_table=False,
            )
        return self._parser

    def parse(self, str_input: str, normalize=True) -> torch.tensor:
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")

        if normalize and self.text_normalizer_call is not None:
            str_input = self.text_normalizer_call(str_input, **self.text_normalizer_call_kwargs)

        if self.learn_alignment:
            eval_phon_mode = contextlib.nullcontext()
            if hasattr(self.tokenizer, "set_phone_prob"):
                eval_phon_mode = self.tokenizer.set_phone_prob(prob=1.0)

            # Disable mixed g2p representation if necessary
            with eval_phon_mode:
                tokens = self.parser(str_input)
        else:
            tokens = self.parser(str_input)

        x = torch.tensor(tokens).unsqueeze_(0).long().to(self.device)
        return x

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "pitch": NeuralType(('B', 'T_audio'), RegressionValuesType()),
            "energy": NeuralType(('B', 'T_audio'), RegressionValuesType(), optional=True),
            "speaker": NeuralType(('B'), Index(), optional=True),
            "pace": NeuralType(optional=True),
            "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
            "attn_prior": NeuralType(('B', 'T_spec', 'T_text'), ProbsType(), optional=True),
            "mel_lens": NeuralType(('B'), LengthsType(), optional=True),
            "input_lens": NeuralType(('B'), LengthsType(), optional=True),
        }
    )
    def forward(
        self,
        *,
        text,
        durs=None,
        pitch=None,
        energy=None,
        speaker=None,
        pace=1.0,
        spec=None,
        attn_prior=None,
        mel_lens=None,
        input_lens=None,
    ):

        mels_pred, *_ = self.fastpitch(
            text=text,
            durs=durs,
            pitch=pitch,
            energy=energy,
            speaker=speaker,
            pace=pace,
            spec=spec,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=input_lens,
        )

        audio_pred = self.hfg_model.generator(x=mels_pred)
        return audio_pred

    @typecheck(output_types={"audio": NeuralType(('B', 'T_audio'), AudioSignal())})
    def convert_text_to_waveform(
        self, tokens: 'torch.tensor', speaker: Optional[int] = None, pace: float = 1.0
    ) -> torch.tensor:
        if self.training:
            logging.warning("convert_text_to_waveform() is meant to be called in eval mode.")
        if isinstance(speaker, int):
            speaker = torch.tensor([speaker]).to(self.device)
        audio = self(text=tokens, durs=None, pitch=None, speaker=speaker, pace=pace)
        return audio

    def training_step(self, batch, batch_idx):
        attn_prior, durs, speaker, energy = None, None, None, None
        if self.learn_alignment:
            assert self.ds_class_name == "TTSDataset", f"Unknown dataset class: {self.ds_class_name}"

            batch_dict = process_batch(batch, self._train_dl.dataset.sup_data_types_set)
            audio = batch_dict.get("audio")
            audio_lens = batch_dict.get("audio_lens")
            text = batch_dict.get("text")
            text_lens = batch_dict.get("text_lens")
            attn_prior = batch_dict.get("align_prior_matrix", None)
            pitch = batch_dict.get("pitch", None)
            energy = batch_dict.get("energy", None)
            speaker = batch_dict.get("speaker_id", None)
            durs = batch_dict.get("durations", None)
        else:
            audio, audio_lens, text, text_lens, attn_prior, pitch, durs, speaker = batch

        mels, spec_len = self.preprocessor(input_signal=audio, length=audio_lens)

        (
            mels_pred,
            dec_lens,
            durs_predicted,
            log_durs_pred,
            pitch_pred,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
            pitch,
            energy_pred,
            energy_tgt,
        ) = self.fp_model(
            text=text,
            durs=durs,
            pitch=pitch,
            energy=energy,
            speaker=speaker,
            pace=1.0,
            spec=mels if self.learn_alignment else None,
            attn_prior=attn_prior,
            mel_lens=spec_len,
            input_lens=text_lens,
        )
        if durs is None:
            durs = attn_hard_dur

        audio = audio.unsqueeze(1)

        fp_inter_mel = self.mel_loss_fn(spect_predicted=mels_pred, spect_tgt=mels)

        # Skip batch if it is too short
        if self.cfg.use_patches:
            ids_str_max = dec_lens - self._cfg.segment_size // self._cfg.n_window_stride + 1
            if len(ids_str_max[ids_str_max < 0]) > 0:
                print('Encountered short batch')
                return

            mels_pred, ids_slice = rand_slice_segments(
                mels_pred, dec_lens, self._cfg.segment_size // self._cfg.n_window_stride
            )
            audio = slice_segments(audio, ids_slice * self.cfg.n_window_stride, self._cfg.segment_size)
            mels = slice_segments(mels, ids_slice, self._cfg.segment_size // self._cfg.n_window_stride)

        audio_pred = self.hfg_model.generator(x=mels_pred)

        audio_pred_mel, _ = self.preprocessor(input_signal=audio_pred.squeeze(1), length=audio_lens)

        optim_g, optim_d = self.optimizers()

        # Train discriminator
        optim_d.zero_grad()

        mpd_score_real, mpd_score_gen, _, _ = self.hfg_model.mpd(y=audio, y_hat=audio_pred.detach())
        loss_disc_mpd, _, _ = self.discriminator_loss(
            disc_real_outputs=mpd_score_real, disc_generated_outputs=mpd_score_gen
        )
        msd_score_real, msd_score_gen, _, _ = self.hfg_model.msd(y=audio, y_hat=audio_pred.detach())
        loss_disc_msd, _, _ = self.discriminator_loss(
            disc_real_outputs=msd_score_real, disc_generated_outputs=msd_score_gen
        )
        loss_d = loss_disc_msd + loss_disc_mpd

        self.manual_backward(loss_d)
        norm_d = clip_grad_value_(
            itertools.chain(self.hfg_model.msd.parameters(), self.hfg_model.mpd.parameters()), None
        )
        optim_d.step()

        # Train generator
        optim_g.zero_grad()
        loss_mel = self.mel_loss_fn(spect_predicted=audio_pred_mel, spect_tgt=mels)
        _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.hfg_model.mpd(y=audio, y_hat=audio_pred)
        _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.hfg_model.msd(y=audio, y_hat=audio_pred)
        loss_fm_mpd = self.feature_loss(fmap_r=fmap_mpd_real, fmap_g=fmap_mpd_gen)
        loss_fm_msd = self.feature_loss(fmap_r=fmap_msd_real, fmap_g=fmap_msd_gen)
        loss_gen_mpd, _ = self.generator_loss(disc_outputs=mpd_score_gen)
        loss_gen_msd, _ = self.generator_loss(disc_outputs=msd_score_gen)
        loss_g = loss_gen_msd + loss_gen_mpd + loss_fm_msd + loss_fm_mpd + loss_mel * self.l1_factor

        if self.cfg.inter_mel_loss:
            loss_g += fp_inter_mel

        dur_loss = self.duration_loss_fn(log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens)
        loss_g += dur_loss
        if self.learn_alignment:
            ctc_loss = self.forward_sum_loss_fn(attn_logprob=attn_logprob, in_lens=text_lens, out_lens=spec_len)
            bin_loss_weight = min(self.current_epoch / self.bin_loss_warmup_epochs, 1.0) * 1.0
            bin_loss = self.bin_loss_fn(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
            loss_g += ctc_loss + bin_loss

        pitch_loss = self.pitch_loss_fn(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
        energy_loss = self.energy_loss_fn(energy_predicted=energy_pred, energy_tgt=energy_tgt, length=text_lens)
        loss_g += pitch_loss + energy_loss

        self.manual_backward(loss_g)
        norm_g = clip_grad_value_(
            itertools.chain(self.hfg_model.generator.parameters(), self.fp_model.parameters()), None
        )
        optim_g.step()

        # Run schedulers
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            sch1, sch2 = schedulers
            sch1.step()
            sch2.step()

        metrics = {
            "g_loss_fm_mpd": loss_fm_mpd,
            "g_loss_fm_msd": loss_fm_msd,
            "g_loss_gen_mpd": loss_gen_mpd,
            "g_loss_gen_msd": loss_gen_msd,
            "total_loss": loss_g,
            "d_loss_mpd": loss_disc_mpd,
            "d_loss_msd": loss_disc_msd,
            "d_loss": loss_d,
            "fp_inter_mel_loss": fp_inter_mel,
            "audio_mel_loss": loss_mel,
            "t_dur_loss": dur_loss,
            "t_pitch_loss": pitch_loss,
            "grad_d": norm_d,
            "grad_g": norm_g,
        }

        if energy_tgt is not None:
            metrics["t_energy_loss"] = energy_loss

        if self.learn_alignment:
            metrics["t_ctc_loss"] = ctc_loss
            metrics["t_bin_loss"] = bin_loss

        self.log_dict(metrics, on_step=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # return
        attn_prior, durs, speaker, energy = None, None, None, None
        if self.learn_alignment:
            assert self.ds_class_name == "TTSDataset", f"Unknown dataset class: {self.ds_class_name}"
            batch_dict = process_batch(batch, self._validation_dl.dataset.sup_data_types_set)
            # print(batch_dict.keys())
            audio = batch_dict.get("audio")
            audio_lens = batch_dict.get("audio_lens")
            text = batch_dict.get("text")
            text_lens = batch_dict.get("text_lens")
            attn_prior = batch_dict.get("align_prior_matrix", None)
            pitch = batch_dict.get("pitch", None)
            energy = batch_dict.get("energy", None)
            speaker = batch_dict.get("speaker_id", None)
        else:
            audio, audio_lens, text, text_lens, durs, pitch, speaker = batch

        mels, mel_lens = self.preprocessor(input_signal=audio, length=audio_lens)

        # Calculate val loss on ground truth durations to better align L2 loss in time
        (
            mels_pred,
            _,
            _,
            log_durs_pred,
            pitch_pred,
            _,
            _,
            _,
            attn_hard_dur,
            pitch,
            energy_pred,
            energy_tgt,
        ) = self.fp_model(
            text=text,
            durs=durs,
            pitch=pitch,
            energy=energy,
            speaker=speaker,
            pace=1.0,
            spec=mels if self.learn_alignment else None,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=text_lens,
        )

        audio_pred = self.hfg_model.generator(x=mels_pred)
        audio_mel_pred, audio_mel_pred_len = self.preprocessor(
            input_signal=audio_pred.squeeze(1), length=audio_lens
        )

        # plot audio once per epoch
        if batch_idx == 0 and isinstance(self.logger, WandbLogger) and HAVE_WANDB:
            logger = self.logger.experiment

            specs = []
            audios = []

            audios += [
                wandb.Audio(
                    audio[0, : audio_lens[0]].data.cpu().to(torch.float).view(-1,).numpy(),
                    caption=f"val_wav_target",
                    sample_rate=self._cfg.sample_rate,
                ),
                wandb.Audio(
                    audio_pred[0, : audio_lens[0]].data.cpu().to(torch.float).view(-1,).numpy(),
                    caption=f"val_wav_predicted",
                    sample_rate=self._cfg.sample_rate,
                ),
            ]

            specs += [
                wandb.Image(
                    plot_spectrogram_to_numpy(mels[0, :, : mel_lens[0]].data.cpu().numpy()), caption=f"val_mel_target",
                ),
                wandb.Image(
                    plot_spectrogram_to_numpy(mels_pred[0, :, : mel_lens[0]].data.cpu().numpy()),
                    caption=f"val_mel_inter",
                ),
                wandb.Image(
                    plot_spectrogram_to_numpy(audio_mel_pred[0, :, : audio_mel_pred_len[0]].data.cpu().numpy()),
                    caption=f"val_mel_predicted",
                ),
            ]

            logger.log({"specs": specs, "audios": audios})

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        if cfg.dataset._target_ == "nemo.collections.tts.data.dataset.TTSDataset":
            phon_mode = contextlib.nullcontext()
            if hasattr(self.tokenizer, "set_phone_prob"):
                phon_mode = self.tokenizer.set_phone_prob(
                    prob=None if name == "val" else self.tokenizer.phoneme_probability
                )

            with phon_mode:
                dataset = instantiate(
                    cfg.dataset,
                    text_normalizer=self.normalizer,
                    text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
                    text_tokenizer=self.tokenizer,
                )
        else:
            dataset = instantiate(cfg.dataset)

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="val")

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []

        return list_of_models

    @property
    def input_types(self):
        return self._input_types

    @property
    def output_types(self):
        return self._output_types
