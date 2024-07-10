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
import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


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
        self.waveform_loss = cfg.get("waveform_loss", False)
        self.cap_vocode = cfg.get("cap_vocode", False)
        self.ce_loss_lambda = 0.1
        self.additional_log_batches = cfg.get("additional_log_batches", 0)
        self.log_media = cfg.get("log_media", True)
        self.silence_value = 0 # -4.5252 for mel

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

    def _download_gigaspeech(self, target_dir, dataset_parts, source="huggingface"):
        """ Download GigaSpeech corpus. """
        if source == "lhotse":
            from lhotse.recipes.gigaspeech import download_gigaspeech, prepare_gigaspeech
            assert hasattr(self._cfg, "password")
            download_gigaspeech(password=self._cfg.password, target_dir=target_dir, dataset_parts=dataset_parts, host="tsinghua")
        elif source == "huggingface":
            import datasets
            from datasets import load_dataset
            for part in dataset_parts:
                part = part.lower()
                if part not in ["xs", "s", "m", "l", "xl"]: # dev/test will auto-downloaded w/ train set
                    continue
                ds = load_dataset("esb/datasets", "gigaspeech", subconfig=part, download_config=datasets.DownloadConfig(resume_download=True))
                print(ds)

    def _prepare_gigaspeech(self, corpus_dir, output_dir, textgrid_dir, dataset_parts, source="huggingface"):
        if source == "lhotse":
            from lhotse.recipes.gigaspeech import download_gigaspeech, prepare_gigaspeech
            from lhotse import CutSet

            logging.info(f"mkdir -p {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            gigaspeech_punctuations = ['<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>']
            gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']

            for subset in dataset_parts:
                manifest_path = os.path.join(output_dir, f"gigaspeech_cuts_{subset}.speech.jsonl.gz")
                if manifest_path not in [self._cfg.train_ds.manifest_filepath, self._cfg.validation_ds.manifest_filepath, self._cfg.test_ds.manifest_filepath]:
                    continue
                if not os.path.exists(manifest_path):
                    # prepare or load recordings/supervisions manifest
                    manifest = prepare_gigaspeech(corpus_dir=corpus_dir, dataset_parts=subset, output_dir=output_dir, num_jobs=self._cfg.ds_kwargs.num_workers)
                    
                    # turn into CutSet
                    manifest = manifest[subset]
                    cuts = CutSet.from_manifests(
                        recordings=manifest["recordings"],
                        supervisions=manifest["supervisions"],
                        output_path=None
                    )

                    # remove punctuations
                    for punctuation in gigaspeech_punctuations:
                        cuts = cuts.transform_text(lambda text: ' '.join(text.replace(punctuation, '').strip().split()))
                    # filter non-speech
                    cuts = cuts.filter_supervisions(lambda s: s.text not in gigaspeech_garbage_utterance_tags)

                    # trim cuts according to supervision segments
                    cuts = cuts.trim_to_supervisions(keep_overlapping=False)
                    
                    logging.info(f"Writing {subset} subset.")
                    cuts.to_file(manifest_path)
                
                else:
                    logging.info(f"Skipping fix, {subset} subset exists.")
        elif source == "huggingface":
            import datasets
            from datasets import load_dataset, Audio
            from lhotse import RecordingSet, Recording, AudioSource, SupervisionSegment, SupervisionSet, CutSet, fix_manifests, validate_recordings_and_supervisions
            from lhotse.recipes.utils import manifests_exist

            def has_valid_audio(ex):
                """ For filtering examples.

                Usage:
                    ds = ds.cast_column("audio", Audio(decode=False))
                    ds = ds.filter(has_valid_audio)
                    ds = ds.cast_column("audio", Audio(decode=True))
                """
                try:
                    assert ex["text"] != ""
                    sf.read(ex["audio"]["path"])
                except Exception:
                    return False
                return True

            def invalid_speech_as_none(batch):
                """ For turning invalid examples into None, to prevent extra file loading during filtering.

                Usage:
                    ds = ds.cast_column("audio", Audio(decode=False))
                    ds = ds.with_transform(invalid_speech_as_none)
                """
                audios = []
                
                for audio in batch["audio"]:
                    try:
                        # since gigaspeech is mono-channel, no need to do extra array processing.
                        array, sampling_rate = sf.read(audio["path"])
                        audio = {
                            "path": audio["path"],
                            "array": array,
                            "sampling_rate": sampling_rate,
                        }
                    except Exception:
                        audio = None
                    audios.append(audio)
                batch["audio"] = audio
                return batch

            _part = None
            for part in dataset_parts:
                if part.lower() in ["xs", "s", "m", "l", "xl"]: # dev/test will auto-downloaded w/ train set
                    _part = part.lower()
                    break
            # for split in ["train", "validation", "test"]:
            for split in ["train", "validation"]:
                # if split == "train":
                #     part = _part
                if split == "validation":
                    part = "DEV"
                elif split == "test":
                    part = "TEST"
                _part = part.lower()
                output_dir = Path(output_dir)

                logging.info(f"Processing GigaSpeech subset: {part}")
                if manifests_exist(
                    part=part, output_dir=output_dir, prefix="gigaspeech", suffix="speech.jsonl.gz"
                ):
                    logging.info(f"GigaSpeech subset: {part} already prepared - skipping.")

                    if self.additional_log_batches > 0:
                        logging.info(f"Processing GigaSpeech subset: {part}_with_jensen")
                        if manifests_exist(
                            part=f"{part}_with_jensen", output_dir=output_dir, types=["cuts"], prefix="gigaspeech", suffix="speech.jsonl.gz"
                        ):
                            logging.info(f"GigaSpeech subset: {part}_with_jensen already prepared - skipping.")

                        else:
                            manifests = self._prepare_jensen(corpus_dir, output_dir)
                            jensen_cuts = manifests["cuts"]
                            gs_cuts = CutSet.from_jsonl_lazy(output_dir / f"gigaspeech_cuts_{part}.speech.jsonl.gz")
                            cuts = jensen_cuts + gs_cuts
                            cuts.to_file(output_dir / f"gigaspeech_cuts_{part}_with_jensen.speech.jsonl.gz")

                    continue

                ds = load_dataset("esb/datasets", "gigaspeech", subconfig=_part, download_config=datasets.DownloadConfig(resume_download=True))
                print(ds)
                ds = ds.cast_column("audio", Audio(decode=False))
                ds = ds.filter(has_valid_audio)
                ds = ds.cast_column("audio", Audio(decode=True))
                print(ds)

                with RecordingSet.open_writer(
                    output_dir / f"gigaspeech_recordings_{part}.speech.jsonl.gz"
                ) as rec_writer, SupervisionSet.open_writer(
                    output_dir / f"gigaspeech_supervisions_{part}.speech.jsonl.gz"
                ) as sup_writer, CutSet.open_writer(
                    output_dir / f"gigaspeech_cuts_{part}.speech.jsonl.gz"
                ) as cut_writer:
                    for data in tqdm(ds[split]):
                        audio_path = Path(data["audio"]["path"])
                        ds_root = audio_path.parents[3]
                        tg_path = ds_root / "MFA" / audio_path.parts[-3] / audio_path.parts[-2] / (audio_path.stem + ".TextGrid")
                        try:
                            assert os.path.exists(tg_path)
                        except:
                            tqdm.write(f"Missing {tg_path}")
                            tqdm.write(str(data))
                            continue

                        num_samples = data["audio"]["array"].shape[-1]
                        duration = round(num_samples / data["audio"]["sampling_rate"], ndigits=8)

                        recordings = [
                            Recording(
                                id=data["id"],
                                sources=[AudioSource(type='file', channels=[0], source=data["audio"]["path"])],
                                sampling_rate=data["audio"]["sampling_rate"],
                                num_samples=num_samples,
                                duration=duration,
                            )
                        ]
                        segments = [
                            SupervisionSegment(
                                id=data["id"],
                                recording_id=data["id"],
                                start=0,
                                duration=duration,
                                channel=0,
                                text=data["text"],
                                language='English',
                                alignment=parse_mfa_textgrid(f_id=tg_path, seg=None),
                            )
                        ]
                        recordings, segments = fix_manifests(
                            recordings=RecordingSet.from_recordings(recordings),
                            supervisions=SupervisionSet.from_segments(segments),
                        )
                        validate_recordings_and_supervisions(
                            recordings=recordings, supervisions=segments
                        )
                        # Create the cut since most users will need it anyway.
                        # There will be exactly one cut since there's exactly one recording.
                        cuts = CutSet.from_manifests(
                            recordings=recordings, supervisions=segments
                        )
                        # Write the manifests
                        rec_writer.write(recordings[0])
                        for s in segments:
                            sup_writer.write(s)
                        cut_writer.write(cuts[0])

    def _prepare_jensen(self, corpus_dir, output_dir):
        from lhotse import RecordingSet, Recording, AudioSource, SupervisionSegment, SupervisionSet, CutSet, fix_manifests, validate_recordings_and_supervisions
        from lhotse.recipes.utils import manifests_exist

        output_dir = Path(output_dir)

        logging.info(f"Processing subset: Jensen's demo.")
        if manifests_exist(
            part="demo", output_dir=output_dir, prefix="jensen", suffix="speech.jsonl.gz"
        ):
            logging.info(f"Jensen's demo already prepared - skipping.")

        else:
            audio_path = f"{corpus_dir}/2024_GTC_jensen/2024_GTC_jensen.mp3"
            recording = Recording.from_file(audio_path).resample(16000)
            
            tg_path = f"{corpus_dir}/2024_GTC_jensen/2024_GTC_jensen.TextGrid"
            alignments = parse_mfa_textgrid(f_id=tg_path, seg=None)
            
            # durations = [[0, 8.35], [8.21, 16.72], [16.45, 31.2], [30.69, 53.6]]
            segments = [
                SupervisionSegment(
                    id="2024_GTC_jensen-0",
                    recording_id="2024_GTC_jensen",
                    start=0,
                    duration=8.35,
                    channel=0,
                    text="the Transformer was invented we were able to scale large language models at incredible rates effectively doubling every six months",
                    language='English',
                    alignment={
                        "words": alignments["words"][0:26],
                        "phones": alignments["phones"][0:105],
                    }
                ),
                SupervisionSegment(
                    id="2024_GTC_jensen-1",
                    recording_id="2024_GTC_jensen",
                    start=8.21,
                    duration=8.51,
                    channel=0,
                    text="now how is it possible that by doubling every six months if you double the size of the model you double the size of your brain you need twice as much information to go fill it",
                    language='English',
                    alignment={
                        "words": alignments["words"][25:65],
                        "phones": alignments["phones"][104:224],
                    },
                ),
                SupervisionSegment(
                    id="2024_GTC_jensen-2",
                    recording_id="2024_GTC_jensen",
                    start=16.45,
                    duration=14.75,
                    channel=0,
                    text="and so every time you double your parameter count you also have to appropriately increase your training token count the combination of those two numbers becomes the computation scale you have to support",
                    language='English',
                    alignment={
                        "words": alignments["words"][64:108],
                        "phones": alignments["phones"][223:367],
                    },
                ),
                SupervisionSegment(
                    id="2024_GTC_jensen-3",
                    recording_id="2024_GTC_jensen",
                    start=30.69,
                    duration=10.78,
                    channel=0,
                    text="the latest the state-of-the-art open AI model is approximately one point eight trillion parameters one point eight trillion parameters required several trillion tokens to go train",
                    language='English',
                    alignment={
                        "words": alignments["words"][107:143],
                        "phones": alignments["phones"][366:486],
                    },
                ),
                SupervisionSegment(
                    id="2024_GTC_jensen-4",
                    recording_id="2024_GTC_jensen",
                    start=41.19,
                    duration=12.41,
                    channel=0,
                    text="when you multiply the two of them together approximately thirty fourty fifty billion quadrillion floating Point operations per second so you have thirty billion quadrillion",
                    language='English',
                    alignment={
                        "words": alignments["words"][142:179],
                        "phones": alignments["phones"][485:621],
                    },
                ),
                SupervisionSegment(
                    id="2024_GTC_jensen-5",
                    recording_id="2024_GTC_jensen",
                    start=53.35,
                    duration=14.34,
                    channel=0,
                    text="a quadrillion is like a peta and so if you had a peta-flop GPU you would need thirty billion seconds to go compute to go train that model thirty billion seconds is approximately one thousand years",
                    language='English',
                    alignment={
                        "words": alignments["words"][178:226],
                        "phones": alignments["phones"][620:769],
                    },
                ),
                SupervisionSegment(
                    id="2024_GTC_jensen-6",
                    recording_id="2024_GTC_jensen",
                    start=67.4,
                    duration=15.773878,
                    channel=0,
                    text="what we need are bigger gpus we need much much bigger gpus we recognized this early on and we realized that the answer is to put a whole bunch of gpus together",
                    language='English',
                    alignment={
                        "words": alignments["words"][225:],
                        "phones": alignments["phones"][768:],
                    },
                ),
            ]
            recordings, segments = fix_manifests(
                recordings=RecordingSet.from_recordings([recording]),
                supervisions=SupervisionSet.from_segments(segments),
            )
            validate_recordings_and_supervisions(
                recordings=recordings, supervisions=segments
            )
            # Create the cut since most users will need it anyway.
            # There will be exactly one cut since there's exactly one recording.
            cuts = CutSet.from_manifests(
                recordings=recordings, supervisions=segments
            )
            cuts = cuts.trim_to_supervisions(keep_overlapping=False)

            # Write the manifests
            with RecordingSet.open_writer(
                output_dir / f"jensen_recordings_demo.speech.jsonl.gz"
            ) as rec_writer, SupervisionSet.open_writer(
                output_dir / f"jensen_supervisions_demo.speech.jsonl.gz"
            ) as sup_writer, CutSet.open_writer(
                output_dir / f"jensen_cuts_demo.speech.jsonl.gz"
            ) as cut_writer:
                for r in recordings:
                    rec_writer.write(r)
                for s in segments:
                    sup_writer.write(s)
                for c in cuts:
                    cut_writer.write(c)

        manifests = {
            "recordings": RecordingSet.from_jsonl_lazy(output_dir / f"jensen_recordings_demo.speech.jsonl.gz"),
            "supervisions": SupervisionSet.from_jsonl_lazy(output_dir / f"jensen_supervisions_demo.speech.jsonl.gz"),
            "cuts": CutSet.from_jsonl_lazy(output_dir / f"jensen_cuts_demo.speech.jsonl.gz"),
        }
        return manifests

    def prepare_data(self) -> None:
        """ Pytorch Lightning hook.

        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#prepare-data

        The following code is basically for transcribed LibriLight.
        """
        if self._cfg.ds_name == "libriheavy":
            self._download_libriheavy(target_dir=self._cfg.libriheavy_dir, dataset_parts=self._cfg.subsets)
            self._prepare_libriheavy(libriheavy_dir=self._cfg.libriheavy_dir, output_dir=self._cfg.manifests_dir, textgrid_dir=self._cfg.textgrid_dir, dataset_parts=self._cfg.subsets)
        else:
            dataset_parts = [
                self._cfg.validation_ds.subset,
                self._cfg.test_ds.subset,
                self._cfg.train_ds.subset,
            ]
            if self._cfg.ds_name == "libritts":
                self._download_libritts(target_dir=self._cfg.corpus_dir, dataset_parts=dataset_parts)
                self._prepare_libritts(corpus_dir=self._cfg.corpus_dir, output_dir=self._cfg.manifests_dir, textgrid_dir=self._cfg.textgrid_dir, dataset_parts=dataset_parts)
            elif self._cfg.ds_name == "gigaspeech":
                # self._download_gigaspeech(target_dir=self._cfg.corpus_dir, dataset_parts=dataset_parts)
                self._prepare_gigaspeech(corpus_dir=self._cfg.corpus_dir, output_dir=self._cfg.manifests_dir, textgrid_dir=self._cfg.textgrid_dir, dataset_parts=dataset_parts)
                pass

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
        ds_kwargs.update({
            "sampling_rate": self.voicebox.audio_enc_dec.sampling_rate,
            "downsample_factor": self.voicebox.audio_enc_dec.downsample_factor, 
        })

        self.set_world_size(self.trainer)

        return get_lhotse_dataloader_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=LhotseTextToSpeechDataset(
                ds_name= self.cfg.ds_name,
                corpus_dir=self.cfg.corpus_dir,
                old_prefix=self.cfg.old_prefix,
                **ds_kwargs,
            ),
        )
    
    def setup_training_data(self, train_data_config: DictConfig | Dict):
        return EncDecRNNTModel.setup_training_data(self, train_data_config)
    
    def setup_validation_data(self, val_data_config: DictConfig | Dict):
        return EncDecRNNTModel.setup_validation_data(self, val_data_config)

    def setup_test_data(self, test_data_config: DictConfig | Dict):
        return EncDecRNNTModel.setup_test_data(self, test_data_config)

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

    @torch.no_grad()
    def parse_input(self, batch):
        # voicebox's sampling rate
        audio = batch["audio"]
        # audio_lens = batch["audio_lens"]
        mel_lens = batch["mel_lens"]
        # tokens = batch["tokens"]
        # token_lens = batch["token_lens"]
        # texts = batch["texts"]
        # mfa tgt
        # durations = batch.get("durations", None)
        # scaled_durations = batch.get("scaled_durations", None)

        self.voicebox.audio_enc_dec.eval()
        mel = self.voicebox.audio_enc_dec.encode(audio)
        assert mel_lens.max() == mel.shape[1], f"{mel_lens.max()} != {mel.shape[1]}"
        # mel_lens = audio_lens * mel.shape[1] // audio.shape[-1]
        batch.update({
            "mel": mel,
            # "mel_lens": mel_lens,
        })

        # if scaled_durations is not None:
        #     batch.update({
        #         "dp_cond": scaled_durations,
        #         "cum_dur": torch.cumsum(scaled_durations, -1),
        #     })
        # elif durations is not None:
        #     cum_dur = torch.cumsum(durations, -1)
        #     dur_ratio = mel_lens / cum_dur[:, -1]
        #     cum_dur = cum_dur * rearrange(dur_ratio, 'b -> b 1')
        #     cum_dur = torch.round(cum_dur)

        #     dp_cond = torch.zeros_like(cum_dur)
        #     dp_cond[:, 0] = cum_dur[:, 0]
        #     dp_cond[:, 1:] = cum_dur[:, 1:] - cum_dur[:, :-1]

        #     batch.update({
        #         "dp_cond": dp_cond,
        #         "cum_dur": cum_dur,
        #     })

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
        aligned_tokens = batch.get("aligned_tokens", None)

        if aligned_tokens is None:
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
        mel = batch['mel']
        mel_lens = batch['mel_lens']
        mel_mask = get_mask_from_lengths(mel_lens) # (b, t)

        pad_mel = torch.ones_like(mel) * self.silence_value
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

    def mfa_align(self, audio, texts: str, sampling_rate: int):
        """run MFA align then load alignment from textgrid file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            print('Temporary directory created at:', temp_dir)
            # You can create files and directories inside the temporary directory
            os.makedirs(f"{temp_dir}/0", exist_ok=True)
            os.makedirs(f"{temp_dir}/MFA", exist_ok=True)
            temp_audio_path = os.path.join(f"{temp_dir}/0", 'speech.wav')
            temp_text_path = os.path.join(f"{temp_dir}/0", 'speech.lab')
            temp_tg_path = os.path.join(f"{temp_dir}/MFA", '0/speech.TextGrid')

            sf.write(temp_audio_path, audio.cpu(), samplerate=sampling_rate)
            with open(temp_text_path, 'w') as temp_file:
                temp_file.write(texts)

            os.system(f"conda run -n aligner bash -c \"mfa align {temp_dir} english_us_arpa english_us_arpa {temp_dir}/MFA \"")

            alignment = parse_mfa_textgrid(temp_tg_path, seg=None)
        return alignment

    @torch.inference_mode()
    def forward(
        self,
        audio: Tensor | None = None,
        audio_lens: Tensor | None = None,
        texts: Optional[List[str]] = None,
        mel: Tensor | None = None,
        mel_lens: Tensor | None = None,
        phoneme_ids: Optional[Tensor] = None,
        alignments: List[Dict[str, List[AlignmentItem]]] | None = None,
        textgrids: List[str] | None = None,
        edit_from: List[str] | None = None,
        edit_to: List[str] | List[Tuple[str, List[str]]] | None = None,
        steps = 3,
        cond_scale = 1.,
        decode_to_audio = True,
        sample_std = 1.0,
        dp_scale = 1.0,
        ztts = True,
        edit_alignments = None,
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
        from lhotse.dataset.collation import collate_vectors
        self.voicebox.audio_enc_dec.eval()

        if mel is None or mel_lens is None:
            assert audio is not None and audio_lens is not None
            assert audio.ndim == 2
            # assert audio.shape[0] == 1
            # audio_lens = torch.tensor([audio.shape[1]], device=self.device)

            # audio to mel
            mel = self.voicebox.audio_enc_dec.encode(audio)
            mel_lens = audio_lens * mel.shape[1] // audio.shape[-1]

        # mfa align if needed
        if alignments is None:
            if textgrids is None:
                alignments = [self.mfa_align(audio=audio[i], texts=text, sampling_rate=self.voicebox.audio_enc_dec.sampling_rate) for i, text in enumerate(texts)]
            else:
                alignments = [parse_mfa_textgrid(tg, None) for tg in textgrids]

        def parse_dp_input_from_ali(alignment, mel_len, ed_f, ed_t, edit_ali=None):
            alignment = fix_alignment(alignment=alignment)

            # group phone alignments by words
            ori_w2p_alis = map_word_phn_alignment(alignment=alignment)
            ori_w2p_alis = resample_ali(ori_w2p_alis, mel_len.unsqueeze(0))

            if edit_ali is not None:
                edit_ali = fix_alignment(alignment=edit_ali)
                edit_ali = map_word_phn_alignment(alignment=edit_ali)
                edit_ali = resample_ali(edit_ali, sec_to_frames=self.voicebox.audio_enc_dec.sampling_rate / self.voicebox.audio_enc_dec.downsample_factor)

            # edit w2p_alignment, also return new-to-origin mapping for later new_cond construction.
            new_w2p_alis, n2o_mapping = edit_w2p_alignment(w2p_alis=ori_w2p_alis, edit_from=ed_f, edit_to=ed_t, edit_ali=edit_ali)

            # post processing phones, adding word postfix and ghost silence, also return phone-to-phone mapping for later new_cond construction.
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

            # get required duration prediction inputs
            phoneme = [ali.symbol for ali in new_phn_alis]
            dp_cond_mask = torch.tensor([ali.start == -1 for ali in new_phn_alis], device=self.device).bool()
            dp_cond = torch.tensor([ali.duration for ali in new_phn_alis], device=self.device)
            ori_dp_cond = torch.tensor([ali.duration for ali in ori_phn_alis], device=self.device)

            tokens = torch.tensor(self.tokenizer.text_to_ids(phoneme)[0], device=self.device, dtype=torch.long)
            # token_lens = torch.tensor([tokens.shape[0]], dtype=torch.long, device=self.device)
            phoneme_mask = torch.ones_like(tokens)
            return {
                "dp_cond": dp_cond,
                "dp_cond_mask": dp_cond_mask,
                "tokens": tokens,
                "phoneme_mask": phoneme_mask,
                "ori_dp_cond": ori_dp_cond,
                "n2o_mapping": n2o_mapping,
                "ori_p2p_mapping": ori_p2p_mapping,
                "new_p2p_mapping": new_p2p_mapping,
            }

        if edit_alignments is None:
            edit_alignments = [None for _ in range(len(alignments))]
        else:
            edit_alignments = [parse_mfa_textgrid(tg, None) for tg in edit_alignments]
        batch = [parse_dp_input_from_ali(alignment, mel_lens[i], edit_from[i], edit_to[i], edit_alignments[i]) for i, alignment in enumerate(alignments)]

        tokens = [dp_in["tokens"] for dp_in in batch]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long, device=self.device)
        tokens = collate_vectors(tokens, padding_value=0)
        ori_dp_cond = collate_vectors([dp_in["ori_dp_cond"] for dp_in in batch], padding_value=0)
        dp_cond = collate_vectors([dp_in["dp_cond"] for dp_in in batch], padding_value=0)
        dp_cond_mask = collate_vectors([dp_in["dp_cond_mask"] for dp_in in batch], padding_value=0).bool()
        phoneme_mask = collate_vectors([dp_in["phoneme_mask"] for dp_in in batch], padding_value=0).bool()

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
        # cut-and-paste predicted durations and duplicate tokens
        new_dur = torch.where(dp_cond_mask, (dp_outputs["durations"] * dp_scale).round(), dp_cond).int().clamp(min=0)
        new_mel_lens = new_dur.sum(-1)

        # construct new_cond
        # new_cond_mask = torch.ones(aligned_tokens.shape, device=self.device)
        new_cond_mask = get_mask_from_lengths(new_mel_lens)
        self_attn_mask = get_mask_from_lengths(new_mel_lens)
        new_cond = torch.zeros((*new_cond_mask.shape, mel.shape[-1]), device=self.device)

        new_cum_dur = new_dur.cumsum(dim=-1)
        ori_cum_dur = ori_dp_cond.int().cumsum(dim=-1)
        for bi, dp_in in enumerate(batch):
            n2o_mapping = dp_in["n2o_mapping"]
            new_p2p_mapping = dp_in["new_p2p_mapping"]
            ori_p2p_mapping = dp_in["ori_p2p_mapping"]
            for i, j in enumerate(n2o_mapping):
                # new i-th phn to ori j-th phn

                # not preserving
                if j == -1: continue

                # ghost silence mapping
                i = new_p2p_mapping[i]
                j = ori_p2p_mapping[j]

                new_slice = slice(0, new_cum_dur[bi, i].item()) if i == 0 else slice(new_cum_dur[bi, i-1].item(), new_cum_dur[bi, i].item())
                ori_slice = slice(0, ori_cum_dur[bi, j].item()) if j == 0 else slice(ori_cum_dur[bi, j-1].item(), ori_cum_dur[bi, j].item())
                new_cond[bi, new_slice] = mel[bi, ori_slice]
                new_cond_mask[bi, new_slice] = 0

        cond_st_idx = torch.arange(new_cond.shape[1], 0, -1, device=self.device).reshape(1, -1) * new_cond_mask
        cond_st_idx = cond_st_idx.argmax(dim=1)
        cond_ed_idx = torch.arange(new_cond.shape[1], device=self.device).reshape(1, -1) * new_cond_mask
        cond_ed_idx = cond_ed_idx.argmax(dim=1) + 1

        # zero-shot TTS
        def parse_zero_shot_TTS(cond, cond_mask, self_attn_mask, tokens, durs):
            # mel lens
            m_lens = self_attn_mask.sum(-1)

            # tail padding
            new_cond_ = torch.cat([cond, torch.ones_like(cond) * self.silence_value], dim=1)
            self_attn_mask_ = torch.cat([self_attn_mask, torch.zeros_like(self_attn_mask)], dim=1).bool()
            new_cond_mask_ = torch.cat([cond_mask, torch.zeros_like(cond_mask)], dim=1).bool()

            new_self_attn_mask = get_mask_from_lengths(m_lens * 2)
            ztts_mask = new_self_attn_mask & ~self_attn_mask_

            new_cond = new_cond_
            new_cond_mask = new_cond_mask_ | ztts_mask

            new_tokens = torch.cat([tokens, tokens], dim=1)
            new_dur = torch.cat([durs, durs], dim=1)
            aligned_tokens = self.duration_predictor.align_phoneme_ids_with_durations(new_tokens, new_dur)
            return {
                "cond": new_cond,
                "cond_mask": new_cond_mask.bool(),
                "self_attn_mask": new_self_attn_mask.bool(),
                "aligned_tokens": aligned_tokens,
            }

        if ztts:    
            args = parse_zero_shot_TTS(new_cond, new_cond_mask, self_attn_mask, tokens, new_dur)
        else:
            args = {
                "cond": new_cond,
                "cond_mask": new_cond_mask,
                "self_attn_mask": self_attn_mask,
                "aligned_tokens": self.duration_predictor.align_phoneme_ids_with_durations(tokens, new_dur),
            }

        pred = self.cfm_wrapper.sample(
            cond=args["cond"],
            cond_mask=args["cond_mask"],
            aligned_phoneme_ids=args["aligned_tokens"],
            self_attn_mask=args["self_attn_mask"],
            steps=steps,
            cond_scale=cond_scale,
            decode_to_audio=False,
            sample_std=sample_std,
        )
        if self.cap_vocode:
            pred = torch.where(args["cond_mask"][:, :, None], pred, args["cond"])

        if ztts:
            edit_mel = torch.ones_like(new_cond) * self.silence_value
            ztts_mel = torch.ones_like(new_cond) * self.silence_value
            for i in range(len(batch)):
                edit_mel[i, :new_mel_lens[i]] = pred[i, :new_mel_lens[i]]
                ztts_mel[i, :new_mel_lens[i]] = pred[i, new_mel_lens[i]:new_mel_lens[i]*2]
            edit_audio = self.voicebox.audio_enc_dec.decode(edit_mel)
            ztts_audio = self.voicebox.audio_enc_dec.decode(ztts_mel)

        else:
            edit_mel = torch.ones_like(new_cond) * self.silence_value
            for i in range(len(batch)):
                edit_mel[i, :new_mel_lens[i]] = pred[i, :new_mel_lens[i]]
            edit_audio = self.voicebox.audio_enc_dec.decode(edit_mel)
        resyn_audio = self.voicebox.audio_enc_dec.decode(mel)
        if resyn_audio.shape[-1] < audio.shape[-1]:
            resyn_audio = F.pad(resyn_audio, (0, audio.shape[-1]-resyn_audio.shape[-1]))

        hop_size = self.voicebox.audio_enc_dec.downsample_factor
        new_audio_lens = torch.clamp((new_mel_lens-1) * hop_size, max=edit_audio.shape[-1]).long()

        new_cond_st_idx = cond_st_idx * hop_size
        new_cond_ed_idx = torch.clamp(cond_ed_idx * hop_size, max=edit_audio.shape[-1])
        ori_cond_st_idx = new_cond_st_idx
        ori_cond_ed_idx = audio_lens - (new_audio_lens - new_cond_ed_idx)

        if ztts:
            cap_audio = torch.zeros_like(edit_audio)
            for i in range(len(batch)):
                cap_audio[i, :new_cond_st_idx[i]] = audio[i, :ori_cond_st_idx[i]]
                cap_audio[i, new_cond_st_idx[i]:new_cond_ed_idx[i]] = ztts_audio[i, new_cond_st_idx[i]:new_cond_ed_idx[i]]
                cap_audio[i, new_cond_ed_idx[i]:new_audio_lens[i]] = audio[i, ori_cond_ed_idx[i]:audio_lens[i]]

        return {
            "edit_audio": edit_audio,
            "ztts_audio": None if not ztts else ztts_audio,
            "cap_audio": None if not ztts else cap_audio,
            "resyn_audio": resyn_audio,
            "edit_mel": edit_mel,
            "ztts_mel": None if not ztts else ztts_mel,
            "ori_mel": mel,
            "ori_mel_lens": mel_lens,
            "ori_audio_lens": audio_lens,
            "ori_cond_st_idx": ori_cond_st_idx,
            "ori_cond_ed_idx": ori_cond_ed_idx,
            "new_mel_lens": new_mel_lens,
            "new_audio_lens": new_audio_lens,
            "new_cond_st_idx": new_cond_st_idx,
            "new_cond_ed_idx": new_cond_ed_idx,
            "args": args,
        }

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.voicebox, norm_type=2)
        self.log_dict(norms)

    @rank_zero_only
    def log_image(self, key: str, image: Any, step: Optional[int] = None, **kwargs: Any) -> None:
        r"""Log images (numpy arrays, or file paths).

        Args:
            key: The key to be used for logging the image files
            image: The image file path, or numpy array to be logged
            step: The step number to be used for logging the image files
            \**kwargs: Optional kwargs are lists passed to each ``Wandb.Image`` instance (ex: caption, sample_rate).

        Optional kwargs are lists passed to each image (ex: caption, sample_rate).

        """
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_writer = logger.experiment
                tb_writer.add_image(key, image, step, dataformats="HWC")
            
            elif isinstance(logger, WandbLogger):
                import wandb
                from wandb.wandb_run import Run
                if not hasattr(self, "wandb_metrics"):
                    self.wandb_metrics = {}

                wandb_logger: Run = logger.experiment
                kwargs["caption"] = f"step: {step}"
                for k in kwargs:
                    kwargs[k] = [kwargs[k]]

                n = len([image])
                kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]
                metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip([image], kwarg_list)]}
                # logger.log_metrics(metrics, step=step)  # type: ignore[arg-type]
                self.wandb_metrics.update(metrics)

    @rank_zero_only
    def log_audio(self, key: str, audio: Any, step: Optional[int] = None, **kwargs: Any) -> None:
        r"""Log audios (numpy arrays, or file paths).

        Args:
            key: The key to be used for logging the audio files
            audio: The audio file path, or numpy array to be logged
            step: The step number to be used for logging the audio files
            \**kwargs: Optional kwargs are lists passed to each ``Wandb.Audio`` instance (ex: caption, sample_rate).

        Optional kwargs are lists passed to each audio (ex: caption, sample_rate).

        """
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_writer = logger.experiment
                tb_writer.add_audio(key, audio, step, **kwargs)
            
            elif isinstance(logger, WandbLogger):
                import wandb
                from wandb.wandb_run import Run
                if not hasattr(self, "wandb_metrics"):
                    self.wandb_metrics = {}

                wandb_logger: Run = logger.experiment
                audios = [audio]
                kwargs["caption"] = f"step: {step}"
                for k in kwargs:
                    kwargs[k] = [kwargs[k]]

                n = len(audios)
                kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]

                metrics = {key: [wandb.Audio(audio, **kwarg) for audio, kwarg in zip(audios, kwarg_list)]}
                # logger.log_metrics(metrics, step=step)  # type: ignore[arg-type]
                self.wandb_metrics.update(metrics)

    @rank_zero_only
    def log_commit(self, step):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                from wandb.wandb_run import Run

                wandb_logger: Run = logger.experiment
                # wandb_logger.log({}, commit=True)
                logger.log_metrics(self.wandb_metrics, step=step)
                self.wandb_metrics = {}

    def train_dp(self, batch, batch_idx):
        self.duration_predictor.train()

        # voicebox's sampling rate
        audio = batch["audio"]
        audio_lens = batch["audio_lens"]
        mel_lens = batch["mel_lens"]
        tokens = batch["tokens"]
        token_lens = batch["token_lens"]
        texts = batch["texts"]

        # mfa tgt
        durations = batch.get("durations", None)
        scaled_durations = batch.get("scaled_durations", None)
        aligned_tokens = batch.get("aligned_tokens", None)
        dp_cond = batch.get("dp_cond", None)

        dp_inputs = self.duration_predictor.parse_dp_input(
            x1=audio,
            audio_len=audio_lens,
            mel_len=mel_lens,
            durations=durations,
            scaled_durations=scaled_durations,
            phoneme_len=token_lens,
            input_sampling_rate=None,
        )

        # if hasattr(self, "_freeze_cfg") and self._freeze_cfg['is_frozen']["duration_predictor"]:
        #     dp_losses = {}
        #     dp_outputs = {
        #         "aligned_phoneme_ids":  self.duration_predictor.align_phoneme_ids_with_durations(tokens, durations*self.duration_predictor.audio_enc_dec.sampling_rate/self.duration_predictor.audio_enc_dec.downsample_factor)
        #     }
        #     return dp_losses, dp_outputs

        dp_loss, dp_losses, dp_outputs = self.duration_predictor.forward(
            cond=dp_cond,               # might be None
            texts=None,                 # converted to phoneme_ids by dataset
            phoneme_ids=tokens,
            phoneme_len=token_lens,
            phoneme_mask=dp_inputs.get("phoneme_mask"),
            cond_drop_prob=self.cfm_wrapper.cond_drop_prob,
            target=dp_cond,
            cond_mask=None,             # would be generated within
            mel=dp_inputs["mel"],
            mel_len=mel_lens,
            mel_mask=dp_inputs["mel_mask"],
            self_attn_mask=dp_inputs.get("phoneme_mask"),
            return_aligned_phoneme_ids=(aligned_tokens is None),
            calculate_cond=True
        )

        if self.log_media and self.training and self.trainer._logger_connector.should_update_logs:
            plot_id = 0
            x1 = dp_inputs["mel"]
            dp_pred = dp_outputs['durations']

            phns = self.tokenizer.decode(tokens[plot_id].cpu().tolist()).split(' ')
            text = texts[plot_id]

            # self.log_image("train_dp/dur",
            #                plot_alignment_to_numpy(tokens[plot_id], dp_cond[plot_id], dp_pred[plot_id], x1[plot_id].T.detach().cpu().numpy()),
            #                self.global_step)
            self.log_image("train_dp/seg",
                           plot_segment_to_numpy(phns, dp_cond[plot_id], dp_pred[plot_id], x1[plot_id].T.detach().cpu().numpy(), text),
                           self.global_step)
            self.log_image("train_dp/bar",
                           plot_duration_barplot_to_numpy(phns, dp_cond[plot_id], dp_pred[plot_id], text),
                           self.global_step)
                           
            self.log_commit(self.global_step)
            
        return dp_losses, dp_outputs

    def train_vb(self, batch, batch_idx):
        self.voicebox.train()

        # voicebox's sampling rate
        audio = batch["audio"]
        audio_lens = batch["audio_lens"]
        mel_lens = batch["mel_lens"]

        # mfa tgt
        tokens = batch.get("aligned_tokens", None)

        vb_inputs = self.cfm_wrapper.parse_vb_input(
            x1=audio,
            cond=audio,
            mel_len=mel_lens,
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

        if getattr(self.voicebox.audio_enc_dec, "preq_ce", False):
            ce_loss = self.cfm_wrapper.cross_entropy_loss(outputs, audio)
            losses['ce'] = ce_loss

        if self.waveform_loss:
            audio_mask = get_mask_from_lengths(audio_lens)
            waveform_loss = self.cfm_wrapper.waveform_loss(outputs, audio, audio_mask)
            losses['waveform'] = waveform_loss
        
        if self.log_media and self.training and self.trainer._logger_connector.should_update_logs:
            plot_id = 0
            if not self.voicebox.no_diffusion:
                x1, x0, w, pred_dx = outputs['vb']['x1'], outputs['vb']['x0'], outputs['vb']['w'], outputs['vb']['pred']
                cond, cond_mask = outputs['vb']["cond"], outputs['vb']["cond_mask"]
                cond = cond * ~cond_mask
                σ = self.cfm_wrapper.sigma
                pred_x1 = pred_dx + (1 - σ) * x0
                self.log_image("train_vb/x1", plot_spectrogram_to_numpy(x1[plot_id].T.detach().cpu().numpy()), self.global_step)
                self.log_image("train_vb/xt", plot_spectrogram_to_numpy(w[plot_id].T.detach().cpu().numpy()), self.global_step)
                self.log_image("train_vb/cond", plot_spectrogram_to_numpy(cond[plot_id].T.detach().cpu().numpy()), self.global_step)
                self.log_image("train_vb/pred_dx", plot_spectrogram_to_numpy(pred_dx[plot_id].T.detach().cpu().numpy()), self.global_step)
                self.log_image("train_vb/pred_x1", plot_spectrogram_to_numpy(pred_x1[plot_id].T.detach().cpu().numpy()), self.global_step)
            else:
                pred_x1 = outputs['vb']['pred']
                cond, cond_mask = outputs['vb']["cond"], outputs['vb']["cond_mask"]
                x1 = cond
                cond = cond * ~cond_mask
                self.log_image("train_vb/x1", plot_spectrogram_to_numpy(x1[plot_id].T.detach().cpu().numpy()), self.global_step)
                self.log_image("train_vb/cond", plot_spectrogram_to_numpy(cond[plot_id].T.detach().cpu().numpy()), self.global_step)
                self.log_image("train_vb/pred_x1", plot_spectrogram_to_numpy(pred_x1[plot_id].T.detach().cpu().numpy()), self.global_step)

            with torch.no_grad():
                if self.cap_vocode:
                    pred_x1 = torch.where(cond_mask, pred_x1, x1)
                pred_audio = self.voicebox.audio_enc_dec.decode(pred_x1)[plot_id].detach().cpu().numpy()
                recon_audio = self.voicebox.audio_enc_dec.decode(x1)[plot_id].detach().cpu().numpy()
            orig_audio = audio[plot_id].detach().cpu().numpy()
            self.log_audio("train_vb/pred_audio", pred_audio / max(np.abs(pred_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
            self.log_audio("train_vb/recon_audio", recon_audio / max(np.abs(recon_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
            self.log_audio("train_vb/orig_audio", orig_audio / max(np.abs(orig_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
            self.log_commit(self.global_step)

        return losses, outputs
    
    def val_vb(self, batch, batch_idx):
        self.voicebox.train()

        # voicebox's sampling rate
        audio = batch["audio"]
        audio_lens = batch["audio_lens"]
        mel_lens = batch["mel_lens"]

        # mfa tgt
        tokens = batch.get("aligned_tokens", None)
        cuts = batch.get("cuts", None)

        vb_inputs = self.cfm_wrapper.parse_vb_input(
            x1=audio,
            cond=audio,
            mel_len=mel_lens,
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

        if getattr(self.voicebox.audio_enc_dec, "preq_ce", False):
            ce_loss = self.cfm_wrapper.cross_entropy_loss(outputs, audio)
            losses['ce'] = ce_loss
        
        if self.waveform_loss:
            audio_mask = get_mask_from_lengths(audio_lens)
            waveform_loss = self.cfm_wrapper.waveform_loss(outputs, audio, audio_mask)
            losses['waveform'] = waveform_loss
        
        if self.log_media and batch_idx <= self.additional_log_batches:
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
                    decode_to_audio=False,
                )
            
            cond_mask = rearrange(cond_mask, '... -> ... 1')
            cond = cond * ~cond_mask
            pred_x1 = output_audio
            if self.cap_vocode:
                pred_x1 = torch.where(cond_mask, pred_x1, x1)

            for plot_id in range(x1.shape[0]):
                log_id = cuts[plot_id].id
                self.log_image(f"val_vb/{log_id}/x1", plot_spectrogram_to_numpy(x1[plot_id, :mel_lens[plot_id]].T.detach().cpu().numpy()), self.global_step)
                self.log_image(f"val_vb/{log_id}/cond", plot_spectrogram_to_numpy(cond[plot_id, :mel_lens[plot_id]].T.detach().cpu().numpy()), self.global_step)
                self.log_image(f"val_vb/{log_id}/pred_x1", plot_spectrogram_to_numpy(pred_x1[plot_id, :mel_lens[plot_id]].T.detach().cpu().numpy()), self.global_step)

                with torch.no_grad():
                    _pred_audio = self.voicebox.audio_enc_dec.decode(pred_x1[None, plot_id, :mel_lens[plot_id]])[0].detach().cpu().numpy()
                    _recon_audio = self.voicebox.audio_enc_dec.decode(x1[None, plot_id, :mel_lens[plot_id]])[0].detach().cpu().numpy()
                _orig_audio = audio[plot_id, :audio_lens[plot_id]].detach().cpu().numpy()
                self.log_audio(f"val_vb/{log_id}/pred_audio", _pred_audio / max(np.abs(_pred_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
                self.log_audio(f"val_vb/{log_id}/recon_audio", _recon_audio / max(np.abs(_recon_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
                self.log_audio(f"val_vb/{log_id}/orig_audio", _orig_audio / max(np.abs(_orig_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
                # self.log_audio(f"val_vb/{log_id}/pred_audio", _pred_audio / np.sqrt(np.mean(_pred_audio ** 2)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
                # self.log_audio(f"val_vb/{log_id}/orig_audio", _orig_audio / np.sqrt(np.mean(_orig_audio ** 2)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
            self.log_commit(self.global_step)

        return losses, outputs
    
    def training_step(self, batch: List, batch_idx: int) -> STEP_OUTPUT:
        # voicebox's sampling rate
        audio = batch["audio"]

        # dp training
        dp_losses, dp_outputs = self.train_dp(
            batch=batch,
            batch_idx=batch_idx,
        )

        # vb training
        losses, outputs = self.train_vb(
            batch=batch,
            batch_idx=batch_idx,
        )
        losses.update(dp_losses)

        dp_loss = losses.get('dp', 0)
        align_loss = losses.get('align', 0)
        bin_loss = losses.get('bin', 0)
        vb_loss = losses.get('vb', 0)
        wv_loss = losses.get('waveform', 0)
        ce_loss = losses.get('ce', 0) * self.ce_loss_lambda
        tenc_loss = losses.get('text_enc', 0)

        loss = align_loss + bin_loss + dp_loss + vb_loss + wv_loss + ce_loss + tenc_loss

        self.log_dict({f"train_loss/{k}": v for k, v in losses.items()}, sync_dist=True, batch_size=audio.shape[0])
        self.log("train_loss_vb", vb_loss, prog_bar=True, sync_dist=True, batch_size=audio.shape[0])

        return loss
    
    @torch.no_grad()
    def val_vb_0_tts(self, batch: List, batch_idx: int) -> STEP_OUTPUT | None:
        batch = self.parse_input(batch)
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

        ori_audio = batch["audio"]
        ori_audio_lens = batch["audio_lens"]
        gen_audio = self.voicebox.audio_enc_dec.decode(gen_mel)
        gen_audio_lens = torch.clamp(ori_audio_lens, max=gen_audio.shape[-1])

        # eval metrics
        self.log("val_num_sample", ori_audio.shape[0], reduce_fx=torch.sum)

        # logging
        if batch_idx == 0:
            for i in range(ori_mel.shape[0]):
                self.log_image(f"val_vb_0_tts/{i}/ori_mel", plot_spectrogram_to_numpy(ori_mel[i, :ori_mel_lens[i]].T.cpu().numpy()), self.global_step)
                self.log_image(f"val_vb_0_tts/{i}/gen_mel", plot_spectrogram_to_numpy(gen_mel[i, :ori_mel_lens[i]].T.cpu().numpy()), self.global_step)

                _gen_audio = gen_audio[i, :gen_audio_lens[i]].cpu().numpy()
                _ori_audio = ori_audio[i, :ori_audio_lens[i]].cpu().numpy()
                self.log_audio(f"val_vb/{i}/gen_audio", _gen_audio / max(np.abs(_gen_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
                self.log_audio(f"val_vb/{i}/ori_audio", _ori_audio / max(np.abs(_ori_audio)), self.global_step, sample_rate=self.voicebox.audio_enc_dec.sampling_rate)
            self.log_commit(self.global_step)

        return

    def validation_step(self, batch: List, batch_idx: int) -> STEP_OUTPUT | None:
        if self.val_0_tts:
            return self.val_vb_0_tts(batch, batch_idx)

        # voicebox's sampling rate
        audio = batch["audio"]

        # dp training
        dp_losses, dp_outputs = self.train_dp(
            batch=batch,
            batch_idx=batch_idx,
        )

        # vb training
        losses, outputs = self.val_vb(
            batch=batch,
            batch_idx=batch_idx,
        )
        losses.update(dp_losses)

        dp_loss = losses.get('dp', 0)
        align_loss = losses.get('align', 0)
        bin_loss = losses.get('bin', 0)
        vb_loss = losses.get('vb', 0)
        wv_loss = losses.get('waveform', 0)
        ce_loss = losses.get('ce', 0) * self.ce_loss_lambda
        tenc_loss = losses.get('text_enc', 0)

        loss = align_loss + bin_loss + dp_loss + vb_loss + wv_loss + ce_loss + tenc_loss
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
    


def parse_mfa_textgrid(f_id, seg=None):
    """ read alignment from MFA textgrid file
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
    """unify silence or unknown phone/word symbols"""
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
    """group phone alignments according to words"""
    phn_alis: List[AlignmentItem] = alignment["phones"]
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

mfa_en_dict = {}
with open("/root/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict", 'r') as f:
    for line in tqdm(f):
        wrd, _, _, _, _, phns = line.strip().split('\t')
        if wrd not in mfa_en_dict:
            mfa_en_dict[wrd] = phns

def edit_w2p_alignment(w2p_alis=None, edit_from="", edit_to="", edit_ali=None):
    """edit a word from alignment
    Return:
        - new_w2p_alis
        - n2o_mapping: new word position in new_w2p_alis to original word position in w2p_alis. For edited words, no mapped word position, so fill in -1 instead.
    """
    import random
    words = [wrd for wrd, _ in w2p_alis]

    word_edit = ' ' not in edit_from

    if word_edit:
        if edit_from is None:
            edit_from = random.choice([wrd for wrd in words if wrd not in ["<eps>", "<unk>"]])
            edit_to = edit_from
        edit_pos = [i for i, wrd in enumerate(words) if wrd == edit_from]
        edit_pos = [random.choice(edit_pos)]
    else:
        words = [(i, wrd) for i, wrd in enumerate(words) if wrd not in ["<eps>", "<unk>", ""]]
        edit_froms = edit_from.split(' ')
        edit_tos = edit_to.split(' ')
        edit_pos = []
        for i , (j, wrd) in enumerate(words):
            for k in range(len(edit_froms)):
                if words[i+k][1] != edit_froms[k]:
                    break
            else:
                edit_pos = list(range(words[i][0], words[i+k][0]+1))
                break
        assert len(edit_pos) > 0

    new_w2p_alis: List[Tuple[str, List[AlignmentItem]]] = []
    ori_phn_alis = []
    n2o_mapping: List[int] = []
    edited = False
    for i, (wrd, phn_alis) in enumerate(w2p_alis):
        # store for calculate masked interval
        ori_phn_alis += phn_alis

        if i in edit_pos:
            if word_edit:
                assert wrd == edit_from

                if isinstance(edit_to, str):
                    # MFA G2P
                    wrd = edit_to
                    phns = os.popen(f"conda run -n aligner bash -c \"echo '{edit_to}' | mfa g2p -n 1 - english_us_arpa - 2> /dev/null\"").read().split('\t')[1].strip().split(' ')
                elif isinstance(edit_to, Tuple):
                    assert isinstance(edit_to[0], str) and isinstance(edit_to[1], List)
                    wrd, phns = edit_to

                # start=-1 to note masked
                phn_alis = [AlignmentItem(symbol=phn, start=-1, duration=0) for phn in phns]
                new_w2p_alis.append((wrd, phn_alis))
                n2o_mapping += [-1] * len(phn_alis)

            elif edited:
                continue

            elif edit_ali:
                if edit_ali[-1][0] == "<eps>":
                    edit_ali = edit_ali[:-1]
                # edit_ali = edit_ali[2:]
                for wrd, phn_alis in edit_ali:
                    new_w2p_alis.append((wrd, phn_alis))
                    n2o_mapping += [-1] * len(phn_alis)
                edited = True
            else:
                for wrd in edit_tos:
                    if wrd in mfa_en_dict:
                        phns = mfa_en_dict[wrd].split(' ')
                    else:
                        phns = os.popen(f"conda run -n aligner bash -c \"echo '{wrd}' | mfa g2p -n 1 - english_us_arpa - 2> /dev/null\"").read().split('\t')[1].strip().split(' ')
                    # start=-1 to note masked
                    phn_alis = [AlignmentItem(symbol=phn, start=-1, duration=0) for phn in phns]
                    new_w2p_alis.append((wrd, phn_alis))
                    n2o_mapping += [-1] * len(phn_alis)

                edited = True
        else:
            if edited:
                edited = False
            new_w2p_alis.append((wrd, phn_alis))
            n2o_mapping += list(range(len(ori_phn_alis)-len(phn_alis), len(ori_phn_alis)))

    return new_w2p_alis, n2o_mapping

def resample_ali(w2p_alis, mel_lens=None, sec_to_frames=None):
    """resample the time unit from second to number of frames"""
    if sec_to_frames is not None:
        ori_durs = torch.tensor([[ali.duration for wrd, phn_alis in w2p_alis for ali in phn_alis]])
        cum_dur = torch.cumsum(ori_durs, -1)
        cum_dur = cum_dur * sec_to_frames
        cum_dur = torch.round(cum_dur)
    if mel_lens is not None:
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
    """post processing alignment, to add word postfix and ghost silence. Since ghost silence are added, also return new_p2p_mapping.
    Return:
        - new_phn_alis
        - new_p2p_mapping: new phone position to original phone position. Used for later new_cond generation.
    """
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