# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import itertools
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm
import matplotlib.pylab as plt
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.utils import logging, model_utils
from nemo.utils.decorators import experimental
from nemo.core.neural_types.elements import AudioSignal, FloatType, Index, IntType, TokenIndex
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.models.vits import VitsModel
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.tts.models.aligner import AlignerModel
from nemo.collections.tts.modules.voicebox_modules import (
    ConditionalFlowMatcherWrapper,
    VoiceBox,
    DurationPredictor,
    MelVoco,
    EncodecVoco,
    get_mask_from_lengths,
    generate_mask_from_repeats,
    interpolate_1d,
    einsum
)

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from nemo_text_processing.text_normalization.normalize import Normalizer
from naturalspeech2_pytorch.utils.tokenizer import Tokenizer

from nemo.collections.tts.parts.utils.helpers import (
    log_audio_to_tb,
    tacotron2_log_to_tb_func,
    # plot_alignment_to_numpy,
    plot_spectrogram_to_numpy,
    waveglow_log_to_tb_func,
    save_figure_to_numpy,
)


@experimental
class VoiceboxModel(TextToWaveform):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.tokenizer: BaseTokenizer | Tokenizer = None
        self.normalizer: Normalizer = None

        aligner = None
        dp_kwargs = {}
        # self.aligner: AlignerModel = None
        if cfg.get("nemo_aligner") and cfg.nemo_aligner.get("from_pretrained"):
            logging.info(cfg.nemo_aligner._target_)
            logging.info(get_class(cfg.nemo_aligner._target_))
            logging.info(cfg.nemo_aligner.from_pretrained)
            # aligner = AlignerModel.from_pretrained("tts_en_radtts_aligner")
            aligner = get_class(cfg.nemo_aligner._target_).from_pretrained(cfg.nemo_aligner.from_pretrained)
            aligner.freeze()

            self.tokenizer = aligner.tokenizer
            self.normalizer = aligner.normalizer
            self.text_normalizer_call_kwargs = aligner.text_normalizer_call_kwargs
            num_tokens = len(aligner.tokenizer.tokens)

            dp_kwargs.update({
                "tokenizer": self.tokenizer,
                "aligner": aligner
            })

        elif cfg.get("nemo_tokenizer"):
            # setup normalizer
            self.text_normalizer_call = None
            self.text_normalizer_call_kwargs = {}
            AlignerModel._setup_normalizer(self, cfg)

            # setup tokenizer
            AlignerModel._setup_tokenizer(self, cfg)    
            assert self.tokenizer is not None

            num_tokens = len(self.tokenizer.tokens)
            self.tokenizer_pad = self.tokenizer.pad
            dp_kwargs.update({
                "tokenizer": self.tokenizer,
            })

        elif cfg.get("mfa_tokenizer"):
            self.normalizer = None
            self.tokenizer = instantiate(cfg.mfa_tokenizer)
            num_tokens = self.tokenizer.vocab_size
            self.tokenizer_pad = self.tokenizer.pad_id
            dp_kwargs.update({
                "tokenizer": self.tokenizer,
            })

        elif cfg.get("tokenizer"):
            self.normalizer = None
            self.tokenizer = instantiate(cfg.tokenizer)
            num_tokens = self.tokenizer.vocab_size
            self.tokenizer_pad = self.tokenizer.pad_id
            dp_kwargs.update({
                "tokenizer": self.tokenizer,
            })

        super().__init__(cfg=cfg, trainer=trainer)

        # self.audio_enc_dec = instantiate(cfg.audio_enc_dec)
        # self.audio_enc_dec.freeze()

        self.duration_predictor = instantiate(
            cfg.duration_predictor,
            **dp_kwargs,
        )
        self.aligner = aligner

        self.voicebox: VoiceBox = instantiate(
            cfg.voicebox,
            num_cond_tokens=num_tokens
        )
        self.cfm_wrapper: ConditionalFlowMatcherWrapper = instantiate(
            cfg.cfm_wrapper,
            voicebox=self.voicebox,
            duration_predictor=self.duration_predictor,
            torchode_method_klass=get_class(cfg.cfm_wrapper.torchode_method_klass)
        )
        
    def prepare_data(self) -> None:
        """ Pytorch Lightning hook.

        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#prepare-data

        The following code is basically for transcribed LibriLight.
        """
        from lhotse import CutSet
        from lhotse.serialization import load_manifest_lazy_or_eager
        from lhotse.recipes.utils import manifests_exist

        logging.info(f"mkdir -p raw_{self._cfg.manifests_dir}")
        os.makedirs("raw_" + self._cfg.manifests_dir, exist_ok=True)
        for subset in self._cfg.subsets:
            if not manifests_exist(subset, "raw_" + self._cfg.manifests_dir, ["cuts"], "libriheavy"):
                logging.info(f"Downloading {subset} subset.")
                os.system(f"wget -P raw_{self._cfg.manifests_dir} -c https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_{subset}.jsonl.gz")
            else:
                logging.info(f"Skipping download, {subset} subset exists.")

        # fix audio path prefix
        old_prefix="download/librilight"
        def change_prefix(cut):
            old_path = cut.recording.sources[0].source
            new_path = old_path.replace(old_prefix, self._cfg.corpus_dir)
            cut.recording.sources[0].source = new_path
            return cut
        
        def textgrid_exist(cut, subset=None):
            cut_id = cut.id
            spk = cut_id.split('/')[1]
            f_id = f"{self.tokenizer.textgrid_dir}/{subset}/{spk}/{','.join(cut_id.split('/'))}.TextGrid"
            return os.path.exists(f_id)

        def parse_cut_mfa_textgrid(seg, subset=None):
            from textgrid import TextGrid, IntervalTier
            from lhotse.supervision import AlignmentItem, SupervisionSet
            seg_id = seg.id
            spk = seg_id.split('/')[1]
            f_id = f"{self.tokenizer.textgrid_dir}/{subset}/{spk}/{','.join(seg_id.split('/'))}.TextGrid"
            tg = TextGrid()
            tg.read(f_id)
            phn_dur = []
            for tier in tg.tiers:
                if tier.name != "phones":
                    continue
                for interval in tier.intervals:
                    minTime = interval.minTime
                    maxTime = interval.maxTime
                    phoneme = interval.mark
                    if phoneme == "":
                        phoneme = "sil"
                    phn_dur.append(AlignmentItem(symbol=phoneme, start=minTime, duration=maxTime - minTime))
            assert len(phn_dur)
            new_sup_seg = seg.with_alignment("phone", phn_dur)
            if seg.duration != maxTime:
                logging.warning(f"recording length unmatch: cut_dur: {seg.duration:.5f}, ali_end: {maxTime:.5f}")
            return new_sup_seg

        logging.info(f"mkdir -p {self._cfg.manifests_dir}")
        os.makedirs(self._cfg.manifests_dir, exist_ok=True)
        for subset in self._cfg.subsets:
            manifest_path = os.path.join(self._cfg.manifests_dir, f"libriheavy_cuts_{subset}.jsonl.gz")
            if manifest_path not in [self._cfg.train_ds.manifest_filepath, self._cfg.validation_ds.manifest_filepath, self._cfg.test_ds.manifest_filepath]:
                continue
            if not os.path.exists(manifest_path):
                logging.info(f"Loading {subset} subset.")
                cuts = load_manifest_lazy_or_eager("raw_" + manifest_path, CutSet)
                logging.info(f"Filtering {subset} subset.")
                cuts = cuts.filter(lambda c: ',' not in c.id)
                cuts = cuts.filter(lambda c: textgrid_exist(c, subset))
                cuts = cuts.map_supervisions(lambda s: parse_cut_mfa_textgrid(s, subset))
                logging.info(f"Writing {subset} subset.")
                with CutSet.open_writer(
                    manifest_path, overwrite=False
                ) as cut_writer, tqdm(desc=f"Write {subset} subset") as progress:
                    for cut in cuts:
                        if cut_writer.contains(cut.id):
                            continue
                        cut_writer.write(cut)
                        progress.update()
                # cuts.to_file(manifest_path)
                del cuts
            else:
                logging.info(f"Skipping fix, {subset} subset exists.")

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of fit, validate, test, or predict.
        This is called on every process when using DDP.

        Args:
            stage: fit, validate, test or predict
        """
        if stage == 'fit':
            train_deferred_setup = (
                'train_ds' in self._cfg
                and self._cfg.train_ds is not None
                and self._cfg.train_ds.get('defer_setup', False)
            )
            if self.train_dataloader() is None and train_deferred_setup:
                self.setup_training_data(self._cfg.train_ds)

        if stage in ('fit', 'validate'):
            val_deferred_setup = (
                'validation_ds' in self._cfg
                and self._cfg.validation_ds is not None
                and self._cfg.validation_ds.get('defer_setup', False)
            )
            if not self.val_dataloader() and val_deferred_setup:
                self.setup_multiple_validation_data(val_data_config=self._cfg.validation_ds)

        if stage == 'test':
            test_deferred_setup = (
                'test_ds' in self._cfg
                and self._cfg.test_ds is not None
                and self._cfg.test_ds.get('defer_setup', False)
            )
            if not self.test_dataloader() and test_deferred_setup:
                self.setup_multiple_test_data(test_data_config=self._cfg.test_ds)

    def parse(self, str_input: str, **kwargs: Any) -> torch.tensor:
        if self.cfg.get("nemo_tokenizer"):
            assert all([k in ['normalize',] for k in kwargs.keys()])
            return VitsModel.parse(self, text=str_input **kwargs)
        
        tokens = self.tokenizer.text_to_ids(text=str_input)
        return torch.tensor(tokens).long().unsqueeze(0).to(self.device)

    def _setup_dataloader_from_config(self, config: Optional[Dict]) -> DataLoader[Any]:
        """Modified from https://github.com/pzelasko/NeMo/blob/feature/lhotse-integration/nemo/collections/asr/models/hybrid_rnnt_ctc_bpe_models.py#L129
        """
        try:
            from nemo.collections.asr.data.lhotse.dataloader import get_lhotse_dataloader_from_config
        except:
            from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
        from nemo.collections.tts.data.text_to_speech_lhotse import LhotseTextToSpeechDataset

        assert config.get("use_lhotse")

        # Note:
        #    Lhotse Dataset only maps CutSet -> batch of tensors, but does not actually
        #    contain any data or meta-data; it is passed to it by a Lhotse sampler for
        #    each sampler mini-batch.
        ds_kwargs = {}
        for kw in ["normalizer", "text_normalizer_call_kwargs", "tokenizer"]:
            if hasattr(self, kw):
                ds_kwargs[kw] = getattr(self, kw)

        self.set_world_size(self.trainer)

        return get_lhotse_dataloader_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=LhotseTextToSpeechDataset(
                corpus_dir=self.cfg.corpus_dir,
                **ds_kwargs
            ),
        )
    
    def setup_training_data(self, train_data_config: DictConfig | Dict):
        return EncDecRNNTModel.setup_training_data(self, train_data_config)
    
    def setup_validation_data(self, val_data_config: DictConfig | Dict):
        return EncDecRNNTModel.setup_validation_data(self, val_data_config)

    def setup_test_data(self, test_data_config: DictConfig | Dict):
        return EncDecRNNTModel.setup_test_data(self, test_data_config)

    # for inference
    @typecheck(
        input_types={
            "tokens": NeuralType(('B', 'T_text'), TokenIndex()),
            "speakers": NeuralType(('B',), Index(), optional=True),
            "noise_scale": NeuralType(('B',), FloatType(), optional=True),
            "length_scale": NeuralType(('B',), FloatType(), optional=True),
            "noise_scale_w": NeuralType(('B',), FloatType(), optional=True),
            "max_len": NeuralType(('B',), IntType(), optional=True),
        }
    )
    def forward(
        self,
        cond = None,
        texts: Optional[List[str]] = None,
        phoneme_ids: Optional[Tensor] = None,
        cond_mask = None,
        steps = 3,
        cond_scale = 1.,
        decode_to_audio = True,
    ):
        """
        Args

            - cond,

                    reference input audio

            - `texts`: Optional[List[str]] || `phoneme_ids`: Optional[Tensor]

                    input texts

            - cond_mask = None,

                    masking context audio

            - steps

                    ODE solver denoising steps

            - cond_scale
            
                    interpolate scaling for classifier-free inference guidance
        """
        audio = self.cfm_wrapper.sample(
            cond=cond,
            texts=texts,
            phoneme_ids=phoneme_ids,
            cond_mask=cond_mask,
            steps=steps,
            cond_scale=cond_scale,
            decode_to_audio=decode_to_audio
        )
        return audio

    
    def training_step(self, batch: List, batch_idx: int) -> STEP_OUTPUT:
        # voicebox's sampling rate
        audio = batch["audio_24k"]
        audio_lens = batch["audio_lens_24k"]
        tokens = batch["tokens"]
        token_lens = batch["token_lens"]

        # nemo aligner input
        audio_22050 = batch["audio_22050"]
        audio_lens_22050 = batch["audio_lens_22050"]
        tgt_len = audio.shape[1]

        # mfa tgt
        durations = batch.get("durations", None)

        audio_mask = get_mask_from_lengths(audio_lens)
        _, losses, outputs = self.cfm_wrapper.forward(
            x1=audio,
            mask=audio_mask,
            phoneme_ids=tokens,
            phoneme_len=token_lens,
            dp_cond=None,
            durations=durations,
            cond=audio,
            cond_mask=None,
            input_sampling_rate=None
        )

        dp_loss = losses['dp']
        align_loss = losses.get('align', 0)
        bin_loss = losses.get('bin', 0)
        vb_loss = losses['vb']

        loss = align_loss + bin_loss + dp_loss + vb_loss

        self.log_dict({f"train_loss/{k}": v for k, v in losses.items()}, prog_bar=True, sync_dist=True, batch_size=audio.shape[0])

        if (batch_idx % 200) == 0:
            tb_writer = self.logger.experiment
            
            plot_id = 0
            x1, x0, w, pred_dx = outputs['vb']['x1'], outputs['vb']['x0'], outputs['vb']['w'], outputs['vb']['pred']
            cond, cond_mask = outputs['vb']["cond"], outputs['vb']["cond_mask"]
            cond = cond * ~cond_mask
            σ = self.cfm_wrapper.sigma
            pred_x1 = pred_dx + (1 - σ) * x0
            tb_writer.add_image("train_vb/x1", plot_spectrogram_to_numpy(x1[plot_id].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")
            tb_writer.add_image("train_vb/xt", plot_spectrogram_to_numpy(w[plot_id].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")
            tb_writer.add_image("train_vb/cond", plot_spectrogram_to_numpy(cond[plot_id].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")
            tb_writer.add_image("train_vb/pred_dx", plot_spectrogram_to_numpy(pred_dx[plot_id].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")
            tb_writer.add_image("train_vb/pred_x1", plot_spectrogram_to_numpy(pred_x1[plot_id].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")

            pred_audio = self.voicebox.audio_enc_dec.decode(pred_x1)[plot_id].detach().cpu().numpy()
            orig_audio = audio[plot_id].detach().cpu().numpy()
            tb_writer.add_audio("train_vb/pred_audio", pred_audio / max(np.abs(pred_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
            tb_writer.add_audio("train_vb/orig_audio", orig_audio / max(np.abs(orig_audio)), self.global_step, sample_rate=24000)

            # plot_id = -1
            dp_cond, dp_pred = outputs['dp']['cond'], outputs['dp']['durations']
            tb_writer.add_image("train_dp/dur",
                                plot_alignment_to_numpy(tokens[plot_id], dp_cond[plot_id], dp_pred[plot_id], x1[plot_id].T.detach().cpu().numpy()),
                                self.global_step, dataformats="HWC")

            phns = self.tokenizer.decode(tokens[plot_id].cpu().tolist()).split(' ')
            text = batch["texts"][plot_id]
            tb_writer.add_image("train_dp/seg",
                                plot_segment_to_numpy(phns, dp_cond[plot_id], dp_pred[plot_id], x1[plot_id].T.detach().cpu().numpy(), text),
                                self.global_step, dataformats="HWC")

        return loss
    
    def validation_step(self, batch: List, batch_idx: int) -> STEP_OUTPUT | None:
        # voicebox's sampling rate
        audio = batch["audio_24k"]
        audio_lens = batch["audio_lens_24k"]
        tokens = batch["tokens"]
        token_lens = batch["token_lens"]

        # mfa tgt
        durations = batch.get("durations", None)

        audio_mask = get_mask_from_lengths(audio_lens)
        _, losses, outputs = self.cfm_wrapper.forward(
            x1=audio,
            mask=audio_mask,
            phoneme_ids=tokens,
            phoneme_len=token_lens,
            dp_cond=None,
            durations=durations,
            cond=audio,
            cond_mask=None,
            input_sampling_rate=None
        )

        dp_loss = losses['dp']
        align_loss = losses.get('align', 0)
        bin_loss = losses.get('bin', 0)
        vb_loss = losses['vb']

        loss = align_loss + bin_loss + dp_loss + vb_loss
        self.log_dict({f"val_loss/{k}": v for k, v in losses.items()}, prog_bar=True, sync_dist=True, batch_size=audio.shape[0])
        self.log("val_loss_total", loss, prog_bar=True, sync_dist=True, batch_size=audio.shape[0])
        return loss

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        list_of_models = []
        return list_of_models

    @typecheck(
        input_types={"tokens": NeuralType(('B', 'T_text'), TokenIndex(), optional=True),},
        output_types={"audio": NeuralType(('B', 'T_audio'), AudioSignal())},
    )
    def convert_text_to_waveform(self, *, tokens, speakers=None):
        audio = self(tokens=tokens, speakers=speakers)[0].squeeze(1)
        return audio
    

@torch.no_grad()
def plot_alignment_to_numpy(phoneme_ids, durations, predictions, spectrogram):
    import numpy as np
    phn_lens = (durations > 0).sum().item()
    phoneme_ids = phoneme_ids[:phn_lens]
    durations = durations[:phn_lens]
    predictions = predictions[:phn_lens]

    ids = torch.arange(phoneme_ids.shape[0]).float().to(phoneme_ids.device)
    repeat_dur_mask = generate_mask_from_repeats(rearrange(durations.clamp(min=1), 't -> 1 t'))
    aligned_dur = (ids @ repeat_dur_mask.float()).cpu().numpy()[0]
    repeat_pred_mask = generate_mask_from_repeats(rearrange(predictions.clamp(min=1), 't -> 1 t'))
    aligned_pred = (ids @ repeat_pred_mask.float()).cpu().numpy()[0]

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    ax.scatter(
        range(len(aligned_dur)), aligned_dur, alpha=0.5, color='green', marker='+', s=1, label='target',
    )
    ax.scatter(
        range(len(aligned_pred)), aligned_pred, alpha=0.5, color='red', marker='.', s=1, label='predicted',
    )
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Cumulated Duration / Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

@torch.no_grad()
def plot_segment_to_numpy(phoneme_ids, durations, predictions, spectrogram, text=None):
    import numpy as np
    phn_lens = durations.nonzero()[-1].item() + 1
    phoneme_ids = phoneme_ids[:phn_lens]
    durations = durations[:phn_lens].clamp(min=1)
    cum_dur = torch.cumsum(durations, -1).cpu().numpy()
    predictions = predictions[:phn_lens].clamp(min=1)
    cum_pred = torch.cumsum(predictions, -1).cpu().numpy()

    # ignore sil prediction
    for i, phn in enumerate(phoneme_ids):
        if phn == 'sil':
            c_dur = cum_dur[i]
            c_pred = cum_pred[i]
            cum_pred[i:] = cum_pred[i:] - c_pred + c_dur
    
    # layout
    phoneme_ids = [pid if i % 2 else pid+" "*10 for i, pid in enumerate(phoneme_ids)]

    # fig, ax = plt.subplots(figsize=(12, 3))
    fig, ax = plt.subplots(figsize=(32, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')

    ax.set_xticks(cum_dur)
    ax.set_xticklabels(phoneme_ids, rotation = 90, ha="right", fontsize=8)

    ax.vlines(cum_dur, ymin=0.0, ymax=max(ax.get_yticks())/3, colors='green')
    for d, p in zip(cum_dur, cum_pred):
        ax.plot([d,p], [max(ax.get_yticks())/3, max(ax.get_yticks())*2/3], color='blue')
    ax.vlines(cum_pred, ymin=max(ax.get_yticks())*2/3, ymax=max(ax.get_yticks()), colors='red')

    plt.colorbar(im, ax=ax)
    if text is not None:
        plt.xlabel(f"Frames (Green target, Red predicted)\n{text}")
    else:
        plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data