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
import tempfile
import shutil
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm
import matplotlib.pylab as plt
import numpy as np
from textgrid import TextGrid
from lhotse.supervision import AlignmentItem


import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import grad_norm


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
            self.normalizer = Normalizer(
                lang="en", input_case="cased", overwrite_cache=True, cache_dir="data/cache_dir",
            )
            text_normalizer_call_kwargs = {"punct_pre_process": True, "punct_post_process": True}
            self.normalizer_call = lambda x: self.normalizer.normalize(x, **text_normalizer_call_kwargs)

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

        self.duration_predictor: DurationPredictor = instantiate(
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

        self.maybe_init_from_pretrained_checkpoint(cfg=cfg, map_location='cpu')

        self.val_0_tts = cfg.get("val_0_tts", False)

    def _download_libriheavy(self, target_dir, dataset_parts):
        """ Download LibriHeavy manifests. """
        from lhotse.recipes.utils import manifests_exist
        logging.info(f"mkdir -p {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        for subset in dataset_parts:
            if not manifests_exist(subset, target_dir, ["cuts"], "libriheavy"):
                logging.info(f"Downloading {subset} subset.")
                os.system(f"wget -P {target_dir} -c https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_{subset}.jsonl.gz")
            else:
                logging.info(f"Skipping download, {subset} subset exists.")

    def _prepare_libriheavy(self, libriheavy_dir, output_dir, textgrid_dir, dataset_parts):
        """ Filter LibriHeavy manifests, and integrate with MFA alignments from textgrids. """
        from lhotse import CutSet
        from lhotse.serialization import load_manifest_lazy_or_eager
        
        def textgrid_filename(cut, subset=None):
            cut_id = cut.id
            spk = cut_id.split('/')[1]
            f_id = f"{textgrid_dir}/{subset}/{spk}/{','.join(cut_id.split('/'))}.TextGrid"
            return f_id

        def parse_cut_mfa_textgrid(seg, subset=None):
            f_id = textgrid_filename(seg, subset=subset)
            new_sup_seg = parse_mfa_textgrid(f_id=f_id, seg=seg)
            return new_sup_seg

        logging.info(f"mkdir -p {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        for subset in dataset_parts:
            manifest_path = os.path.join(output_dir, f"libriheavy_cuts_{subset}.jsonl.gz")
            ori_manifest_path = os.path.join(libriheavy_dir, f"libriheavy_cuts_{subset}.jsonl.gz")
            if manifest_path not in [self._cfg.train_ds.manifest_filepath, self._cfg.validation_ds.manifest_filepath, self._cfg.test_ds.manifest_filepath]:
                continue
            if not os.path.exists(manifest_path):
                logging.info(f"Loading {subset} subset.")
                cuts = load_manifest_lazy_or_eager(ori_manifest_path, CutSet)
                logging.info(f"Filtering {subset} subset.")
                cuts = cuts.filter(lambda c: ',' not in c.id)
                cuts = cuts.filter(lambda c: os.path.exists(textgrid_filename(c, subset)))
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
        
    def _download_libritts(self, target_dir, dataset_parts):
        """ Download LibriTTS corpus. """
        from lhotse.recipes import download_libritts

        target_dir = Path(target_dir).parent

        download_libritts(target_dir=target_dir, dataset_parts=dataset_parts)

    def _prepare_libritts(self, corpus_dir, output_dir, textgrid_dir, dataset_parts):
        """ Prepare LibriTTS manifests, and integrate with MFA alignments from textgrids. """
        from lhotse.recipes import prepare_libritts
        from lhotse import CutSet
        from lhotse.serialization import load_manifest_lazy_or_eager
        
        def textgrid_filename(cut, subset=None):
            cut_id = cut.id
            spk, chp = cut_id.split('_')[:2]
            f_id = f"{textgrid_dir}/{subset}/{spk}/{chp}/{cut_id}.TextGrid"
            # print(f_id, os.path.exists(f_id))
            return f_id

        def parse_cut_mfa_textgrid(seg, subset=None):
            f_id = textgrid_filename(seg, subset=subset)
            new_sup_seg = parse_mfa_textgrid(f_id=f_id, seg=seg)
            return new_sup_seg

        logging.info(f"mkdir -p {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        for subset in dataset_parts:
            manifest_path = os.path.join(output_dir, f"libritts_cuts_{subset}.jsonl.gz")
            if manifest_path not in [self._cfg.train_ds.manifest_filepath, self._cfg.validation_ds.manifest_filepath, self._cfg.test_ds.manifest_filepath]:
                continue
            if not os.path.exists(manifest_path):
                manifest = prepare_libritts(corpus_dir=corpus_dir, dataset_parts=subset, output_dir=None, num_jobs=self._cfg.ds_kwargs.num_workers, link_previous_utt=True)
                manifest = manifest[subset]
                cuts = CutSet.from_manifests(
                    recordings=manifest["recordings"],
                    supervisions=manifest["supervisions"],
                    output_path=None
                )
                cuts = cuts.modify_ids(lambda cid: cid.rsplit('-', 1)[0])

                logging.info(f"Filtering {subset} subset.")
                cuts = cuts.filter(lambda c: os.path.exists(textgrid_filename(c, subset)))
                cuts = cuts.map_supervisions(lambda s: parse_cut_mfa_textgrid(s, subset))

                logging.info(f"Writing {subset} subset.")
                cuts.to_file(manifest_path)
            
            else:
                logging.info(f"Skipping fix, {subset} subset exists.")

    def prepare_data(self) -> None:
        """ Pytorch Lightning hook.

        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#prepare-data

        The following code is basically for transcribed LibriLight.
        """
        if self._cfg.ds_name == "libriheavy":
            self._download_libriheavy(target_dir=self._cfg.libriheavy_dir, dataset_parts=self._cfg.subsets)
            self._prepare_libriheavy(libriheavy_dir=self._cfg.libriheavy_dir, output_dir=self._cfg.manifests_dir, textgrid_dir=self._cfg.textgrid_dir, dataset_parts=self._cfg.subsets)
        elif self._cfg.ds_name == "libritts":
            def get_subset(manifest_filepath):
                return '_'.join(manifest_filepath.split('/')[-1].split('.')[0].split('_')[2:])

            dataset_parts = [
                subset for subset in self._cfg.subsets 
                if subset in [
                    get_subset(self._cfg.train_ds.manifest_filepath),
                    get_subset(self._cfg.validation_ds.manifest_filepath),
                    get_subset(self._cfg.test_ds.manifest_filepath)
                ]
            ]
            self._download_libritts(target_dir=self._cfg.corpus_dir, dataset_parts=dataset_parts)
            self._prepare_libritts(corpus_dir=self._cfg.corpus_dir, output_dir=self._cfg.manifests_dir, textgrid_dir=self._cfg.textgrid_dir, dataset_parts=dataset_parts)

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

    def parse(self, text: str, normalize=True) -> torch.tensor:
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")

        # normalize
        if normalize and self.text_normalizer_call is not None:
            text = self.text_normalizer_call(text, **self.text_normalizer_call_kwargs)

        # phonemize
        text = os.popen("conda run -n aligner bash -c \"echo '...' | mfa g2p -n 1 - english_us_arpa -\"").read().split('\t')[1].strip()

        # tokenize
        tokens = self.tokenizer.text_to_ids(text)[0]

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
        ds_kwargs = config.get("ds_kwargs", {})
        ds_kwargs = OmegaConf.to_container(ds_kwargs, resolve=True)
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

    def mfa_align(self, audio, texts: str, sampling_rate: int):
        with tempfile.TemporaryDirectory() as temp_dir:
            print('Temporary directory created at:', temp_dir)
            # You can create files and directories inside the temporary directory
            temp_audio_path = os.path.join(temp_dir, 'speech.wav')
            temp_text_path = os.path.join(temp_dir, 'speech.lab')
            temp_tg_path = os.path.join(temp_dir, 'speech.TextGrid')

            sf.write(temp_audio_path, audio, samplerate=sampling_rate, format='WAV')
            with open(temp_text_path, 'w') as temp_file:
                temp_file.write(texts)

            os.system(f"conda run -n aligner bash -c \"mfa align_one {temp_audio_path} {temp_text_path} english_us_arpa english_us_arpa {temp_tg_path} \"")

            alignment = parse_mfa_textgrid(temp_tg_path, seg=None)
        return alignment

    # for inference
    @torch.inference_mode()
    def forward(
        self,
        audio: Tensor = None,
        texts: Optional[List[str]] = None,
        phoneme_ids: Optional[Tensor] = None,
        alignment = None,
        edit_from: List[str] | None = None,
        edit_to: List[str] | None = None,
        steps = 3,
        cond_scale = 1.,
        decode_to_audio = True,
    ):
        """
        Args:

        Input speech-text pair:

            - `cond`, reference input audio
            - `texts` or `phoneme_ids`, input texts

        Edit:
            - `alignment` = None, dictionary of MFA textgrid, including phoneme/word durations
            - `cond_mask` = None, masking context audio
            - `edit_from` and `edit_to`, words to edit

        Generation:
            - `steps`, ODE solver denoising steps
            - `cond_scale`, interpolate scaling for classifier-free inference guidance
        """
        assert audio.ndim == 2
        assert audio.shape[0] == 1
        audio_lens = torch.tensor([audio.shape[1]], device=self.device)

        self.voicebox.audio_enc_dec.eval()
        mel = self.voicebox.audio_enc_dec.encode(audio)
        mel_lens = audio_lens * mel.shape[1] // audio.shape[-1]

        if alignment is None:
            alignment = self.mfa_align(audio=audio, texts=texts, sampling_rate=24000)
        alignment = fix_alignment(alignment=alignment)
        ori_w2p_alis = map_word_phn_alignment(alignment=alignment)
        ori_w2p_alis = resample_ali(ori_w2p_alis, mel_lens)
        new_w2p_alis, n2o_mapping = edit_w2p_alignment(w2p_alis=ori_w2p_alis, edit_from=edit_from, edit_to=edit_to)

        # post processing phones
        ori_phn_alis, ori_p2p_mapping = process_alignment(
            w2p_alis=ori_w2p_alis,
            use_word_postfix=self.cfg.ds_kwargs.use_word_postfix,
            use_word_ghost_silence=self.cfg.ds_kwargs.use_word_ghost_silence,
        )
        new_phn_alis, new_p2p_mapping = process_alignment(
            w2p_alis=new_w2p_alis,
            use_word_postfix=self.cfg.ds_kwargs.use_word_postfix,
            use_word_ghost_silence=self.cfg.ds_kwargs.use_word_ghost_silence,
        )
        phoneme = [ali.symbol for ali in new_phn_alis]
        dp_cond_mask = torch.tensor([[ali.start == -1 for ali in new_phn_alis]], device=self.device).bool()
        dp_cond = torch.tensor([[ali.duration for ali in new_phn_alis]], device=self.device)
        ori_dp_cond = torch.tensor([[ali.duration for ali in ori_phn_alis]], device=self.device)

        tokens = torch.tensor([self.tokenizer.text_to_ids(phoneme)[0]], device=self.device)
        token_lens = torch.tensor([tokens.shape[1]], dtype=torch.long, device=self.device)
        phoneme_mask = torch.ones_like(tokens).bool()

        self.duration_predictor.eval()
        dp_outputs = self.duration_predictor.forward(
            cond=dp_cond,
            texts=None,
            phoneme_ids=tokens,
            phoneme_len=token_lens,
            phoneme_mask=phoneme_mask,
            # cond_drop_prob=self.cfm_wrapper.cond_drop_prob,
            target=dp_cond,
            cond_mask=dp_cond_mask,
            self_attn_mask=phoneme_mask,
            return_aligned_phoneme_ids=False,
        )
        new_dur = torch.where(dp_cond_mask, dp_outputs["durations"].round(), dp_cond).int().clamp(min=0)
        aligned_tokens = self.duration_predictor.align_phoneme_ids_with_durations(tokens, new_dur)
        new_mel_lens = new_dur.sum().reshape(1,)

        new_cond = torch.zeros((1, new_mel_lens.item(), mel.shape[-1]), device=self.device)
        new_cond_mask = torch.ones((1, new_mel_lens.item()), device=self.device)
        new_cum_dur = new_dur.cumsum(dim=-1)
        ori_cum_dur = ori_dp_cond.int().cumsum(dim=-1)
        for i, j in enumerate(n2o_mapping):
            # new i-th phn to ori j-th phn
            if j == -1: 
                # not preserving
                continue
            # ghost silence mapping
            i = new_p2p_mapping[i]
            j = ori_p2p_mapping[j]

            new_slice = slice(0, new_cum_dur[0, i].item()) if i == 0 else slice(new_cum_dur[0, i-1].item(), new_cum_dur[0, i].item())
            ori_slice = slice(0, ori_cum_dur[0, j].item()) if j == 0 else slice(ori_cum_dur[0, j-1].item(), ori_cum_dur[0, j].item())
            new_cond[0, new_slice] = mel[0, ori_slice]
            new_cond_mask[0, new_slice] = 0
        self_attn_mask = torch.ones_like(new_cond_mask)

        audio = self.cfm_wrapper.sample(
            cond=mel,
            cond_mask=new_cond_mask.bool(),
            aligned_phoneme_ids=aligned_tokens,
            self_attn_mask=self_attn_mask.bool(),
            steps=steps,
            cond_scale=cond_scale,
            decode_to_audio=decode_to_audio
        )
        return audio

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.voicebox, norm_type=2)
        self.log_dict(norms)

    def train_dp(self, audio, audio_mask, tokens, token_lens, texts, durations, batch_idx):
        self.duration_predictor.train()

        dp_inputs = self.duration_predictor.parse_dp_input(
            x1=audio,
            mask=audio_mask,
            durations=durations,
            phoneme_len=token_lens,
            input_sampling_rate=None,
        )

        dp_loss, dp_losses, dp_outputs = self.duration_predictor.forward(
            cond=dp_inputs.get("dp_cond"),               # might be None
            texts=None,                 # converted to phoneme_ids by dataset
            phoneme_ids=tokens,
            phoneme_len=token_lens,
            phoneme_mask=dp_inputs.get("phoneme_mask"),
            cond_drop_prob=self.cfm_wrapper.cond_drop_prob,
            target=dp_inputs.get("dp_cond"),
            cond_mask=None,             # would be generated within
            mel=dp_inputs["mel"],
            mel_len=dp_inputs["mel_len"],
            mel_mask=dp_inputs["mel_mask"],
            self_attn_mask=dp_inputs.get("phoneme_mask"),
            return_aligned_phoneme_ids=True,
            calculate_cond=True
        )
        dp_outputs["cond"] = dp_inputs.get("dp_cond")

        if self.training and self.trainer._logger_connector.should_update_logs:
            tb_writer = self.logger.experiment
            
            plot_id = 0
            x1 = dp_inputs["mel"]
            dp_cond, dp_pred = dp_outputs['cond'], dp_outputs['durations']
            # tb_writer.add_image("train_dp/dur",
            #                     plot_alignment_to_numpy(tokens[plot_id], dp_cond[plot_id], dp_pred[plot_id], x1[plot_id].T.detach().cpu().numpy()),
            #                     self.global_step, dataformats="HWC")

            phns = self.tokenizer.decode(tokens[plot_id].cpu().tolist()).split(' ')
            text = texts[plot_id]
            tb_writer.add_image("train_dp/seg",
                                plot_segment_to_numpy(phns, dp_cond[plot_id], dp_pred[plot_id], x1[plot_id].T.detach().cpu().numpy(), text),
                                self.global_step, dataformats="HWC")
            tb_writer.add_image("train_dp/bar",
                                plot_duration_barplot_to_numpy(phns, dp_cond[plot_id], dp_pred[plot_id], text),
                                self.global_step, dataformats="HWC")
            
        return dp_losses, dp_outputs

    def val_vb(self, audio, audio_mask, tokens, batch_idx):
        self.voicebox.train()

        vb_inputs = self.cfm_wrapper.parse_vb_input(
            x1=audio,
            mask=audio_mask,
            cond=audio,
            input_sampling_rate=None
        )

        x1 = vb_inputs['x1']
        cond = vb_inputs['cond']
        self_attn_mask = vb_inputs['mask']

        _, losses, outputs = self.cfm_wrapper.forward(
            x1=x1,
            mask=self_attn_mask,
            phoneme_ids=tokens,
            cond=cond,
            cond_mask=None,
            input_sampling_rate=None
        )
        
        if batch_idx == 0:
            # first batch of validation
            self.voicebox.eval()
            cond_mask = self.voicebox.create_cond_mask(
                batch=cond.shape[0],
                seq_len=cond.shape[1],
                cond_token_ids=tokens,
                self_attn_mask=self_attn_mask,
                training=True,
                frac_lengths_mask=(0.1, 0.5),
            )
            if self.voicebox.no_diffusion:
                output_audio = self.cfm_wrapper.forward(
                    x1=x1,
                    mask=self_attn_mask,
                    phoneme_ids=tokens,
                    cond=cond,
                    cond_mask=cond_mask,
                    input_sampling_rate=None
                )
            else:
                output_audio = self.cfm_wrapper.sample(
                    cond=cond,
                    self_attn_mask=self_attn_mask,
                    aligned_phoneme_ids=tokens,
                    cond_mask=cond_mask,
                    steps=100,
                    decode_to_audio=False
                )
            
            audio_len = audio_mask.sum(-1)
            mel_len = self_attn_mask.sum(-1)

            cond = cond * ~rearrange(cond_mask, '... -> ... 1')
            pred_x1 = output_audio

            tb_writer = self.logger.experiment
            
            for plot_id in range(x1.shape[0]):
                tb_writer.add_image(f"val_vb/{plot_id}/x1", plot_spectrogram_to_numpy(x1[plot_id, :mel_len[plot_id]].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")
                tb_writer.add_image(f"val_vb/{plot_id}/cond", plot_spectrogram_to_numpy(cond[plot_id, :mel_len[plot_id]].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")
                tb_writer.add_image(f"val_vb/{plot_id}/pred_x1", plot_spectrogram_to_numpy(pred_x1[plot_id, :mel_len[plot_id]].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")

                _pred_audio = self.voicebox.audio_enc_dec.decode(pred_x1[None, plot_id, :mel_len[plot_id]])[0].detach().cpu().numpy()
                _orig_audio = audio[plot_id, :audio_len[plot_id]].detach().cpu().numpy()
                tb_writer.add_audio(f"val_vb/{plot_id}/pred_audio", _pred_audio / max(np.abs(_pred_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
                tb_writer.add_audio(f"val_vb/{plot_id}/orig_audio", _orig_audio / max(np.abs(_orig_audio)), self.global_step, sample_rate=24000)
                # tb_writer.add_audio(f"val_vb/{plot_id}/pred_audio", _pred_audio / np.sqrt(np.mean(_pred_audio ** 2)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
                # tb_writer.add_audio(f"val_vb/{plot_id}/orig_audio", _orig_audio / np.sqrt(np.mean(_orig_audio ** 2)), self.global_step, sample_rate=24000)

        return losses, outputs

    def train_vb(self, audio, audio_mask, tokens, batch_idx):
        vb_inputs = self.cfm_wrapper.parse_vb_input(
            x1=audio,
            mask=audio_mask,
            cond=audio,
            input_sampling_rate=None
        )

        self.voicebox.train()

        _, losses, outputs = self.cfm_wrapper.forward(
            x1=vb_inputs['x1'],
            mask=vb_inputs['mask'],
            phoneme_ids=tokens,
            cond=vb_inputs['cond'],
            cond_mask=None,
            input_sampling_rate=None
        )
        
        if self.training and self.trainer._logger_connector.should_update_logs:
            tb_writer = self.logger.experiment
            
            plot_id = 0
            x1, x0, w, pred_dx = outputs['vb']['x1'], outputs['vb']['x0'], outputs['vb']['w'], outputs['vb']['pred']
            cond, cond_mask = outputs['vb']["cond"], outputs['vb']["cond_mask"]
            cond = cond * ~cond_mask
            σ = self.cfm_wrapper.sigma
            pred_x1 = pred_dx + (1 - σ) * x0 if not self.voicebox.no_diffusion else pred_dx
            tb_writer.add_image("train_vb/x1", plot_spectrogram_to_numpy(x1[plot_id].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")
            tb_writer.add_image("train_vb/xt", plot_spectrogram_to_numpy(w[plot_id].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")
            tb_writer.add_image("train_vb/cond", plot_spectrogram_to_numpy(cond[plot_id].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")
            tb_writer.add_image("train_vb/pred_dx", plot_spectrogram_to_numpy(pred_dx[plot_id].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")
            tb_writer.add_image("train_vb/pred_x1", plot_spectrogram_to_numpy(pred_x1[plot_id].T.detach().cpu().numpy()), self.global_step, dataformats="HWC")

            pred_audio = self.voicebox.audio_enc_dec.decode(pred_x1)[plot_id].detach().cpu().numpy()
            orig_audio = audio[plot_id].detach().cpu().numpy()
            tb_writer.add_audio("train_vb/pred_audio", pred_audio / max(np.abs(pred_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
            tb_writer.add_audio("train_vb/orig_audio", orig_audio / max(np.abs(orig_audio)), self.global_step, sample_rate=24000)

        return losses, outputs
    
    @torch.no_grad()
    def parse_input(self, batch):
        # voicebox's sampling rate
        audio = batch["audio_24k"]
        audio_lens = batch["audio_lens_24k"]
        tokens = batch["tokens"]
        token_lens = batch["token_lens"]
        texts = batch["texts"]
        # mfa tgt
        durations = batch.get("durations", None)

        self.voicebox.audio_enc_dec.eval()
        mel = self.voicebox.audio_enc_dec.encode(audio)
        mel_lens = audio_lens * mel.shape[1] // audio.shape[-1]
        batch.update({
            "mel": mel,
            "mel_lens": mel_lens,
        })

        if durations is not None:
            cum_dur = torch.cumsum(durations, -1)
            dur_ratio = mel_lens / cum_dur[:, -1]
            cum_dur = cum_dur * rearrange(dur_ratio, 'b -> b 1')
            cum_dur = torch.round(cum_dur)

            dp_cond = torch.zeros_like(cum_dur)
            dp_cond[:, 0] = cum_dur[:, 0]
            dp_cond[:, 1:] = cum_dur[:, 1:] - cum_dur[:, :-1]

            batch.update({
                "dp_cond": dp_cond,
                "cum_dur": cum_dur,
            })

        return batch

    @torch.no_grad()
    def parse_val_vb_input(self, batch):
        batch = self.parse_input(batch)
        mel = batch['mel']
        mel_lens = batch['mel_lens']
        mel_mask = get_mask_from_lengths(mel_lens) # (b, t)
        batch.update({
            "mel_mask": mel_mask,
        })

        tokens = batch['tokens']
        durations = batch['dp_cond']
        cum_dur = batch['cum_dur']

        aligned_tokens = self.duration_predictor.align_phoneme_ids_with_durations(tokens, durations)
        batch.update({
            "aligned_tokens": aligned_tokens
        })

        self.voicebox.eval()
        cond_mask = self.voicebox.create_cond_mask(
            batch=mel.shape[0],
            seq_len=mel.shape[1],
            cond_token_ids=aligned_tokens,
            self_attn_mask=mel_mask,
            training=True,
            frac_lengths_mask=(0.1, 0.5),
        )
        batch.update({
            "cond": mel,
            "cond_mask": cond_mask,
            "self_attn_mask": mel_mask,
        })

        return batch

    @torch.no_grad()
    def parse_0_tts(self, batch):
        batch = self.parse_input(batch)
        mel = batch['mel']
        mel_lens = batch['mel_lens']
        mel_mask = get_mask_from_lengths(mel_lens) # (b, t)

        pad_mel = torch.ones_like(mel) * -4.5252
        new_mel = torch.cat([mel, pad_mel], dim=1)
        new_mask = get_mask_from_lengths(mel_lens * 2)

        pad_mask = torch.zeros_like(mel_mask)
        ori_mask = torch.cat([mel_mask, pad_mask], dim=1).bool()
        cond_mask = new_mask & ~ori_mask
        batch.update({
            "cond": new_mel,
            "cond_mask": cond_mask,
            "self_attn_mask": new_mask,
        })

        tokens = batch['tokens']
        durations = batch['dp_cond']
        cum_dur = batch['cum_dur']

        new_tokens = torch.cat([tokens, tokens], dim=1)
        new_dur = torch.cat([durations, durations], dim=1)
        new_aligned_tokens = self.duration_predictor.align_phoneme_ids_with_durations(new_tokens, new_dur)
        batch.update({
            "aligned_tokens": new_aligned_tokens
        })
        return batch

    @torch.no_grad()
    def val_vb_0_tts(self, batch: List, batch_idx: int) -> STEP_OUTPUT | None:
        batch = self.parse_0_tts(batch)

        self.voicebox.eval()

        cond = batch['cond']
        self_attn_mask = batch['self_attn_mask']
        aligned_tokens = batch['aligned_tokens']
        cond_mask = batch['cond_mask']

        out_spec = self.cfm_wrapper.sample(
            cond=cond,
            self_attn_mask=self_attn_mask,
            aligned_phoneme_ids=aligned_tokens,
            cond_mask=cond_mask,
            steps=10,
            decode_to_audio=False
        )

        ori_mel = batch['mel']
        ori_mel_lens = batch['mel_lens']
        gen_idx = torch.arange(ori_mel.shape[1]).to(ori_mel.device).reshape(1, -1, 1).expand_as(ori_mel)
        gen_idx = gen_idx + ori_mel_lens.reshape(-1, 1, 1)
        gen_mel = torch.gather(out_spec, 1, gen_idx)
        # gen_mel = torch.zeros_like(ori_mel)
        # for i in range(ori_mel.shape[0]):
        #     gen_mel[i, :] = out_spec[i, ori_mel_lens[i]:ori_mel_lens[i]+gen_mel.shape[1]]

        ori_audio = batch["audio_24k"]
        ori_audio_lens = batch["audio_lens_24k"]
        gen_audio = self.voicebox.audio_enc_dec.decode(gen_mel)
        gen_audio_lens = torch.clamp(ori_audio_lens, max=gen_audio.shape[-1])

        # eval metrics
        self.log("val_num_sample", ori_audio.shape[0], reduce_fx=torch.sum)

        # logging
        if batch_idx == 0:
            tb_writer = self.logger.experiment
            for i in range(ori_mel.shape[0]):
                tb_writer.add_image(f"val_vb_0_tts/{i}/ori_mel", plot_spectrogram_to_numpy(ori_mel[i, :ori_mel_lens[i]].T.cpu().numpy()), self.global_step, dataformats="HWC")
                tb_writer.add_image(f"val_vb_0_tts/{i}/gen_mel", plot_spectrogram_to_numpy(gen_mel[i, :ori_mel_lens[i]].T.cpu().numpy()), self.global_step, dataformats="HWC")

                _gen_audio = gen_audio[i, :gen_audio_lens[i]].cpu().numpy()
                _ori_audio = ori_audio[i, :ori_audio_lens[i]].cpu().numpy()
                tb_writer.add_audio(f"val_vb/{i}/gen_audio", _gen_audio / max(np.abs(_gen_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
                tb_writer.add_audio(f"val_vb/{i}/ori_audio", _ori_audio / max(np.abs(_ori_audio)), self.global_step, sample_rate=24000)

        return

    def training_step(self, batch: List, batch_idx: int) -> STEP_OUTPUT:
        # voicebox's sampling rate
        audio = batch["audio_24k"]
        audio_lens = batch["audio_lens_24k"]
        tokens = batch["tokens"]
        token_lens = batch["token_lens"]
        texts = batch["texts"]

        # # nemo aligner input
        # audio_22050 = batch["audio_22050"]
        # audio_lens_22050 = batch["audio_lens_22050"]
        # tgt_len = audio.shape[1]

        # mfa tgt
        durations = batch.get("durations", None)

        audio_mask = get_mask_from_lengths(audio_lens)

        # dp training
        dp_losses, dp_outputs = self.train_dp(
            audio=audio,
            audio_mask=audio_mask,
            tokens=tokens,
            token_lens=token_lens,
            texts=texts,
            durations=durations,
            batch_idx=batch_idx,
        )

        # vb training
        losses, outputs = self.train_vb(
            audio=audio,
            audio_mask=audio_mask,
            tokens=dp_outputs.get("aligned_phoneme_ids"),
            batch_idx=batch_idx,
        )
        losses.update(dp_losses)

        dp_loss = losses['dp']
        align_loss = losses.get('align', 0)
        bin_loss = losses.get('bin', 0)
        vb_loss = losses['vb']

        loss = align_loss + bin_loss + dp_loss + vb_loss

        self.log_dict({f"train_loss/{k}": v for k, v in losses.items()}, sync_dist=True, batch_size=audio.shape[0])
        self.log("train_loss_vb", vb_loss, prog_bar=True, sync_dist=True, batch_size=audio.shape[0])

        return loss
    
    def validation_step(self, batch: List, batch_idx: int) -> STEP_OUTPUT | None:
        if self.val_0_tts:
            return self.val_vb_0_tts(batch, batch_idx)

        # voicebox's sampling rate
        audio = batch["audio_24k"]
        audio_lens = batch["audio_lens_24k"]
        tokens = batch["tokens"]
        token_lens = batch["token_lens"]
        texts = batch["texts"]

        # mfa tgt
        durations = batch.get("durations", None)

        audio_mask = get_mask_from_lengths(audio_lens)

        # dp training
        dp_losses, dp_outputs = self.train_dp(
            audio=audio,
            audio_mask=audio_mask,
            tokens=tokens,
            token_lens=token_lens,
            texts=texts,
            durations=durations,
            batch_idx=batch_idx,
        )

        # vb training
        losses, outputs = self.val_vb(
        # losses, outputs = self.train_vb(
            audio=audio,
            audio_mask=audio_mask,
            tokens=dp_outputs.get("aligned_phoneme_ids"),
            batch_idx=batch_idx,
        )
        losses.update(dp_losses)

        dp_loss = losses['dp']
        align_loss = losses.get('align', 0)
        bin_loss = losses.get('bin', 0)
        vb_loss = losses['vb']

        loss = align_loss + bin_loss + dp_loss + vb_loss
        self.log_dict({f"val_loss/{k}": v for k, v in losses.items()}, sync_dist=True, batch_size=audio.shape[0])
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
    


def parse_mfa_textgrid(f_id, seg: None):
    """
    Args
        - f_id: textgrid_filename
        - seg: `cut.supervisions.segment`, this function can be used for updating `cut.supervisions.segment`.
    """
    tg = TextGrid()
    tg.read(f_id)
    alignment = {}
    if seg is not None:
        new_sup_seg = seg

    for tier in tg.tiers:
        _dur = []
        for interval in tier.intervals:
            minTime = interval.minTime
            maxTime = interval.maxTime
            mark = interval.mark
            if tier.name == "phones":
                if mark == "" or  mark == "sp":
                    mark = "sil"
            # elif tier.name == "words":
            #     if mark in ["", "sil", "<eps>"]:
            #         mark = "<eps>"
            #     elif mark in ["spn", "<unk>"]:
            #         mark = "<unk>"
            _dur.append(AlignmentItem(symbol=mark, start=minTime, duration=maxTime - minTime))
        assert len(_dur)

        # update cut segment
        alignment[tier.name] = _dur
        if seg is not None:
            new_sup_seg = new_sup_seg.with_alignment(tier.name, _dur)
        
        # warning
        if f"{seg.duration:.2f}" != f"{maxTime:.2f}":
            logging.warning(f"recording length unmatch: cut_dur: {seg.duration:.2f}, ali_end: {maxTime:.2f}")

    if seg is not None:
        return new_sup_seg
    return alignment

def fix_alignment(alignment: Dict[str, List[AlignmentItem]]) -> Dict[str, List[AlignmentItem]]:
    def phn_transform(symbol):
        if symbol == "" or  symbol == "sp":
            symbol = "sil"
        return symbol

    def wrd_transform(symbol):
        if symbol in ["", "sil", "<eps>"]:
            symbol = "<eps>"
        elif symbol in ["spn", "<unk>"]:
            symbol = "<unk>"
        return symbol

    new_alignment = {}
    for name in alignment:
        if name == "phones":
            new_alignment["phones"] = [ali.transform(phn_transform) for ali in alignment[name]]
        elif name == "words":
            new_alignment["words"] = [ali.transform(wrd_transform) for ali in alignment[name]]
    return new_alignment

def map_word_phn_alignment(alignment: Dict[str, List[AlignmentItem]]):
    phn_alis: List[AlignmentItem] = alignment["phones"]
    phn_dur: List[Tuple[str, float]] = []
    for ali in phn_alis:
        phn_dur.append((ali.symbol, ali.duration))

    word_alis: List[AlignmentItem] = alignment["words"]
    w2p_alis: List[Tuple[str, List[AlignmentItem]]] = []
    phn_id = 0
    for ali in word_alis:
        wrd = ali.symbol
        w2p_alis.append((wrd, []))

        wrd_st = ali.start
        wrd_ed = wrd_st + ali.duration

        phn_st = phn_alis[phn_id].start
        phn_ed = phn_st + phn_alis[phn_id].duration
        while phn_st >= wrd_st and phn_ed <= wrd_ed:
            w2p_alis[-1][-1].append(phn_alis[phn_id])

            phn_id += 1
            if phn_id >= len(phn_alis):
                break
            phn_st = phn_alis[phn_id].start
            phn_ed = phn_st + phn_alis[phn_id].duration

    return w2p_alis

def edit_w2p_alignment(w2p_alis=None, edit_from="", edit_to=""):
    new_w2p_alis: List[Tuple[str, List[AlignmentItem]]] = []
    n2o_mapping: List[int] = []
    for i, (wrd, phn_alis) in enumerate(w2p_alis):
        if wrd == edit_from:
            # store for calculate masked interval
            ori_phn_alis += phn_alis

            # MFA G2P
            phns = os.popen(f"conda run -n aligner bash -c \"echo '{edit_to}' | mfa g2p -n 1 - english_us_arpa -\"").read().split('\t')[1].strip().split(' ')
            
            # start=-1 to note masked
            wrd = edit_to
            phn_alis = [AlignmentItem(symbol=phn, start=-1, duration=0) for phn in phns]

            new_w2p_alis.append((wrd, phn_alis))
            n2o_mapping += [-1] * len(phn_alis)
        else:
            new_w2p_alis.append((wrd, phn_alis))
            n2o_mapping += list(range(len(ori_phn_alis)-len(phn_alis), len(ori_phn_alis)))

    return new_w2p_alis, n2o_mapping

def resample_ali(w2p_alis, mel_lens):
    ori_durs = torch.tensor([[ali.duration for wrd, phn_alis in w2p_alis for ali in phn_alis]], device=mel_lens.device)
    cum_dur = torch.cumsum(ori_durs, -1)
    dur_ratio = mel_lens / cum_dur[:, -1]
    cum_dur = cum_dur * rearrange(dur_ratio, 'b -> b 1')
    cum_dur = torch.round(cum_dur)

    rsmp_durs = torch.zeros_like(cum_dur)
    rsmp_durs[:, 0] = cum_dur[:, 0]
    rsmp_durs[:, 1:] = cum_dur[:, 1:] - cum_dur[:, :-1]
    pos = 0
    _end = 0
    rsmp_w2p_alis = []
    for wrd, phn_alis in w2p_alis:
        rsmp_w2p_alis.append((wrd, []))
        for ali in phn_alis:
            rsmp_w2p_alis[-1][-1].append(AlignmentItem(symbol=ali.symbol, start=_end, duration=cum_dur[0, pos].item() - _end))
            _end = cum_dur[0, pos].item()
            pos += 1
    return rsmp_w2p_alis


def process_alignment(w2p_alis: List[Tuple[str, List[AlignmentItem]]]=None, use_word_postfix=True, use_word_ghost_silence=True, edit_from="", edit_to=""):
    new_phn_alis: List[AlignmentItem] = []
    new_p2p_mapping: List[int] = []
    _wrd = "<eps>"
    for wrd, phn_alis in w2p_alis:
        if use_word_ghost_silence and len(new_phn_alis) > 0:
            if wrd != "<eps>" and _wrd != "<eps>":
                # symbol
                sil_symbol = "sil_S" if use_word_postfix else "sil"
                # start
                assert len(new_phn_alis) > 0
                _start = new_phn_alis[-1].start
                start_ = phn_alis[0].start
                if _start == -1 or start_ == -1:
                    start = -1
                else:
                    start = start_
                # append
                new_phn_alis.append(AlignmentItem(symbol=sil_symbol, start=start, duration=0))
            _wrd = wrd

        postfixs = [""] * len(phn_alis)
        if use_word_postfix:
            if len(phn_alis) == 1:
                postfixs = ["_S"]
            else:
                postfixs = ["_B"] + ["_I"] * (len(phn_alis)-2) + ["_E"]

        for p_ali, postfix in zip(phn_alis, postfixs):
            new_p2p_mapping.append(len(new_phn_alis))
            new_phn_alis.append(AlignmentItem(symbol=p_ali.symbol + postfix, start=p_ali.start, duration=p_ali.duration))

    return new_phn_alis, new_p2p_mapping


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
    durations = durations[:phn_lens].clamp(min=0)
    cum_dur = torch.cumsum(durations, -1).cpu().numpy()
    predictions = predictions[:phn_lens].clamp(min=0)
    cum_pred = torch.cumsum(predictions, -1).cpu().numpy()

    # ignore sil prediction
    for i, phn in enumerate(phoneme_ids):
        if phn in ['sil', 'sil_S']:
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


@torch.no_grad()
def plot_duration_barplot_to_numpy(phoneme_ids, durations, predictions, text=None):
    import numpy as np
    phn_lens = durations.nonzero()[-1].item() + 1
    phoneme_ids = phoneme_ids[:phn_lens]
    durations = durations[:phn_lens].clamp(min=0).cpu().numpy()
    predictions = predictions[:phn_lens].clamp(min=0).cpu().numpy()

    # fig, ax = plt.subplots(figsize=(12, 3))
    fig, ax = plt.subplots(figsize=(48, 3))

    # for plots
    x = np.arange(phn_lens)
    width = 0.4
    rects = ax.bar(x - width/2, durations, width, label="target")
    # ax.bar_label(rects, padding=3)
    rects = ax.bar(x + width/2, predictions, width, label="predict")
    # ax.bar_label(rects, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(phoneme_ids, rotation = 90)
    ax.set_xlim(-1, phn_lens)
    ax.set_ylim(top=30)
    ax.legend(loc='upper left', ncols=3)
    ax2 = ax.twiny()
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{d:<2.0f}/{p:<2.0f}" for d, p in zip(durations, predictions)], rotation=90)
    ax2.set_xlim(-1, phn_lens)

    if text is not None:
        plt.xlabel(f"# of Phonemes\n{text}")
    else:
        plt.xlabel("# of Phonemes")
    plt.ylabel("# of Frames")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data