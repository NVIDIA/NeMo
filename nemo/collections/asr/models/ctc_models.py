# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import json
import os
import tempfile
from math import ceil
from typing import Any, Dict, List, Optional, Union

import editdistance
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import (
    _AudioTextDataset,
    cache_datastore_manifests,
    expand_sharded_filepaths,
)
from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRModuleMixin, ASRTranscriptionMixin, InterCTCMixin, TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import GenericTranscriptionType, TranscriptionReturnType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.asr.parts.utils.ipl_utils import *
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.parts.preprocessing.parsers import make_parser
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging

__all__ = ['EncDecCTCModel']


class EncDecCTCModel(ASRModel, ExportableEncDecModel, ASRModuleMixin, InterCTCMixin, ASRTranscriptionMixin):
    """Base class for encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        if cfg.get("ipl", None):
            with open_dict(cfg.ipl):
                cfg.ipl.num_all_files, cfg.ipl.num_cache_files = count_files_for_pseudo_labeling(
                    cfg.ipl.manifest_filepath, cfg.ipl.get('dataset_weights', None)
                )
                if not cfg.train_ds.get("is_tarred", False):
                    if not cfg.ipl.get("cache_manifest", None):
                        cfg.ipl.cache_manifest = formulate_cache_manifest_names(
                            cfg.ipl.manifest_filepath, cfg.ipl.cache_prefix, is_tarred=False
                        )
                else:
                    cfg.ipl.cache_manifest = []
                    cfg.ipl.all_cache_manifests = formulate_cache_manifest_names(
                        cfg.ipl.manifest_filepath, cfg.ipl.cache_prefix, is_tarred=True
                    )

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = EncDecCTCModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = EncDecCTCModel.from_config_dict(self._cfg.encoder)

        with open_dict(self._cfg):
            if "feat_in" not in self._cfg.decoder or (
                not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self._cfg.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.decoder.num_classes < 1 and self.cfg.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.decoder.num_classes, len(self.cfg.decoder.vocabulary)
                    )
                )
                cfg.decoder["num_classes"] = len(self.cfg.decoder.vocabulary)

        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)

        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecCTCModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = CTCDecoding(self.cfg.decoding, vocabulary=OmegaConf.to_container(self.decoder.vocabulary))

        # Setup metric with decoding strategy
        self.wer = WER(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        # setting up interCTC loss (from InterCTCMixin)
        self.setup_interctc(decoder_name='decoder', loss_name='loss', wer_name='wer')

        # Adapter modules setup (from ASRAdapterModelMixin)
        self.setup_adapters()

    def on_fit_start(self):
        """
        Cache datastore manifests for non tarred unlabeled data for pseudo labeling.
        This function prevents caching audio files at the end of every epoch.
        """
        if self.cfg.get("ipl"):
            if not self.cfg.ipl.get("is_tarred", False):
                cache_datastore_manifests(self.cfg.ipl.get("manifest_filepath"), cache_audio=True)
        super().on_fit_start()

    def on_train_epoch_end(self):
        """
        This function is mainly used for iterative pseudo labeling algorithm.
        To make it work in config file 'ipl' parameters should be provided.

        """
        if not self.cfg.get("ipl"):
            return

        if self.cfg.ipl.m_epochs > 0:
            self.cfg.ipl.m_epochs -= 1
            return
        needs_update = True

        if self.cfg.ipl.m_epochs == 0:
            self.build_cache(update_whole_cache=True)

            self.encoder.set_dropout(self.cfg.ipl.dropout)
            self.cfg.ipl.m_epochs -= 1
            needs_update = False

        if self.cfg.ipl.m_epochs == -1 and self.cfg.ipl.n_l_epochs > 0:
            self.cfg.ipl.n_l_epochs -= 1
        else:
            if needs_update:
                self.build_cache(update_whole_cache=False)

            if self.cfg.train_ds.get("is_tarred", False):
                final_cache_manifests = self.combine_cache_hypotheses()

            if self.cfg.ipl.n_l_epochs == 0:
                if self.cfg.train_ds.get("is_tarred", False):
                    if isinstance(self.cfg.ipl.tarred_audio_filepaths, str):
                        if isinstance(self.cfg.train_ds.tarred_audio_filepaths, str):
                            self.cfg.train_ds.tarred_audio_filepaths = [
                                [self.cfg.train_ds.tarred_audio_filepaths],
                                [self.cfg.ipl.tarred_audio_filepaths],
                            ]
                        else:
                            self.cfg.train_ds.tarred_audio_filepaths.append([self.cfg.ipl.tarred_audio_filepaths])
                    else:
                        if isinstance(self.cfg.train_ds.tarred_audio_filepaths, str):
                            self.cfg.train_ds.tarred_audio_filepaths = [[self.cfg.train_ds.tarred_audio_filepaths]]
                        self.cfg.train_ds.tarred_audio_filepaths += self.cfg.ipl.tarred_audio_filepaths

                    if isinstance(self.cfg.train_ds.manifest_filepath, str):
                        self.cfg.train_ds.manifest_filepath = [[self.cfg.train_ds.manifest_filepath]]

                    self.cfg.train_ds.manifest_filepath += final_cache_manifests

                else:
                    if isinstance(self.cfg.train_ds.manifest_filepath, str):
                        self.cfg.train_ds.manifest_filepath = [self.cfg.train_ds.manifest_filepath]
                        self.cfg.train_ds.manifest_filepath.append(self.cfg.ipl.cache_manifest)
                    else:
                        self.cfg.train_ds.manifest_filepath.append(self.cfg.ipl.cache_manifest)

                self.cfg.ipl.n_l_epochs = -1
                self.trainer.reload_dataloaders_every_n_epochs = 1
                if self.cfg.ipl.get("limit_train_batches", None):
                    self.trainer.limit_train_batches = self.cfg.ipl["limit_train_batches"]

            torch.distributed.barrier()

            if self.cfg.train_ds.get("use_lhotse", False):
                self.setup_training_data(self.cfg.train_ds, do_caching=False, update_limit_train_batches=False)
            else:
                self.setup_training_data(self.cfg.train_ds, do_caching=False, update_limit_train_batches=True)

    def build_cache(self, update_whole_cache: bool):
        """
        Function to build cache file for maintaining pseudo labels.
        Args:
            update_whole_cache: (bool) Indicates whether to update the entire cache or only a portion of it based on sampling.
        """
        if self.cfg.train_ds.get("is_tarred", False):

            if update_whole_cache:
                self.create_tar_cache_hypotheses(self.cfg.ipl.manifest_filepath, self.cfg.ipl.tarred_audio_filepaths)
            else:
                self.update_tar_cache_hypotheses(self.cfg.ipl.all_cache_manifests, self.cfg.ipl.tarred_audio_filepaths)
        else:
            self.create_cache_hypotheses(self.cfg.ipl.manifest_filepath, update_whole_cache)

    def create_cache_hypotheses(self, manifests: Union[List[List[str]], str], update_whole_cache: bool = True):
        """
        Function to create cache file for unlabeled dataset
        Args:
            update_whole_cache: Indicates whether to update the entire cache or only a portion of it based on sampling.
            manifests:  Manifest file(s) from which pseudo labels will be generated
        """

        whole_pseudo_data = []
        update_data = []

        manifest_paths = [manifests] if isinstance(manifests, str) else manifests
        dataset_weights = self.cfg.ipl.get("dataset_weights", [1] * len(manifest_paths))

        if not isinstance(dataset_weights, ListConfig) and not isinstance(dataset_weights, List):
            dataset_weights = [float(dataset_weights)]

        for idx, manifest_path in enumerate(manifest_paths):
            manifest_data = process_manifest(manifest_path)
            whole_pseudo_data.extend(manifest_data)
            weight = dataset_weights[idx] if idx < len(dataset_weights) else 1
            update_data.extend(sample_data(manifest_data, weight, update_whole_cache, self.cfg.ipl.p_cache))

        with tempfile.TemporaryDirectory() as tmpdir:
            temporary_manifest = os.path.join(tmpdir, f'manifest_{torch.distributed.get_rank()}.json')
            with open(temporary_manifest, 'w', encoding='utf-8') as temp_manifest:
                transcriptions = [data_entry.get('text', "") for data_entry in update_data]
                for data_entry in update_data:
                    json.dump(data_entry, temp_manifest, ensure_ascii=False)
                    temp_manifest.write('\n')

            hypotheses = self.generate_pseudo_labels(
                temporary_manifest,
                target_transcripts=transcriptions,
                restore_pc=self.cfg.ipl.restore_pc,
                batch_size=self.cfg.ipl.batch_size,
            )
        torch.distributed.barrier()
        gathered_hypotheses = [None] * torch.distributed.get_world_size()
        gathered_data = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_data, update_data)
        torch.distributed.all_gather_object(gathered_hypotheses, hypotheses)
        if torch.distributed.get_rank() == 0:
            write_cache_manifest(self.cfg.ipl.cache_manifest, gathered_hypotheses, gathered_data, update_whole_cache)
        torch.distributed.barrier()

    def create_tar_cache_hypotheses(
        self, manifests: Union[List[List[str]], str], tarred_audio_filepaths: Union[List[List[str]], str]
    ):
        """
        Function to create cache file for tarred unlabeled dataset for the first time
        Args:
            manifests:  Manifest file(s) from which pseudo labels will be generated
            tarred_audio_filepaths: Tar file paths for tarred datasets
        """
        if isinstance(manifests, str):
            manifests = [[manifests]]

        if isinstance(tarred_audio_filepaths, str):
            tarred_audio_filepaths = [[tarred_audio_filepaths]]

        self.cfg.ipl.cache_manifest = []
        for manifest, tarred_audio_filepath in zip(manifests, tarred_audio_filepaths):
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.distributed.barrier()

                expanded_audio = expand_sharded_filepaths(
                    tarred_audio_filepath[0],
                    shard_strategy='scatter',
                    world_size=self.world_size,
                    global_rank=self.global_rank,
                )
                expand_manifests = expand_sharded_filepaths(
                    manifest[0], shard_strategy='scatter', world_size=self.world_size, global_rank=self.global_rank
                )
                number_of_manifests = len(expand_manifests)

                shard_manifest_data = []
                cache_manifest = []
                transcriptions = []
                for idx, manifest_path in enumerate(expand_manifests):

                    manifest_data = process_manifest(manifest_path)
                    shard_manifest_data.append(manifest_data)

                    base_path, filename = os.path.split(manifest_path)
                    cache_file = os.path.join(base_path, f'{self.cfg.ipl.cache_prefix}_cache_{filename}')
                    cache_manifest.append(cache_file)

                    temporary_manifest = os.path.join(tmpdir, f'temp_{filename}')

                    with open(temporary_manifest, 'w', encoding='utf-8') as temp_manifest:

                        for data_entry in manifest_data:

                            if not data_entry.get("text", None):
                                data_entry['text'] = ""
                            transcriptions.append(data_entry.get('text', ""))
                            json.dump(data_entry, temp_manifest, ensure_ascii=False)
                            temp_manifest.write('\n')

                if number_of_manifests > 1:

                    temporary_manifest, expanded_audio = handle_multiple_tarr_filepaths(
                        filename, tmpdir, number_of_manifests, expanded_audio[0]
                    )
                else:
                    expanded_audio = expanded_audio[0]

                if self.cfg.train_ds.get("is_tarred", False):
                    hypotheses = self.generate_pseudo_labels(
                        cache_manifest=temporary_manifest,
                        tarred_audio_filepaths=expanded_audio,
                        target_transcripts=None,
                        restore_pc=self.cfg.ipl.restore_pc,
                        batch_size=self.cfg.ipl.batch_size,
                    )

                else:
                    hypotheses = self.generate_pseudo_labels(
                        manifest,
                        target_transcripts=None,
                        restore_pc=self.cfg.ipl.restore_pc,
                        batch_size=self.cfg.ipl.batch_size,
                    )

                write_tarr_cache_manifest(
                    cache_manifest,
                    update_data=shard_manifest_data,
                    hypotheses=hypotheses,
                    use_lhotse=self.cfg.train_ds.get('use_lhotse', False),
                )
                torch.distributed.barrier()
                self.cfg.ipl.cache_manifest.append(cache_manifest)

    def update_tar_cache_hypotheses(
        self, manifests: Union[List[List[str]], str], tarred_audio_filepaths: Union[List[List[str]], str]
    ):
        """
        With given probability randomly chooses part of the cache hypotheses, generates new pseudo labels for them and updates the cache.
        Args:
            manifests: Cache manifest files where pseudo labels are kept.
            tarred_audio_filepaths: Path to tarred audio files.
        """
        if isinstance(manifests, str):
            manifests = [[manifests]]

        if isinstance(tarred_audio_filepaths, str):
            tarred_audio_filepaths = [[tarred_audio_filepaths]]

        torch.distributed.barrier()
        for manifest, tarred_audio_filepath in zip(manifests, tarred_audio_filepaths):
            with tempfile.TemporaryDirectory() as tmpdir:
                expanded_audio = expand_sharded_filepaths(
                    tarred_audio_filepath[0],
                    shard_strategy='scatter',
                    world_size=self.world_size,
                    global_rank=self.global_rank,
                )
                manifest = expand_sharded_filepaths(
                    manifest[0], shard_strategy='scatter', world_size=self.world_size, global_rank=self.global_rank
                )
                shard_manifest_data = []
                number_of_manifests = len(manifest)
                all_indices = []
                for _, manifest_path in enumerate(manifest):

                    manifest_data = process_manifest(manifest_path)
                    update_size = int(len(manifest_data) * self.cfg.ipl.p_cache)
                    random.seed()
                    indices = random.sample(range(len(manifest_data)), update_size)
                    update_data = [manifest_data[index] for index in indices]
                    shard_manifest_data.append(manifest_data)

                    all_indices.append(indices)
                    _, filename = os.path.split(manifest_path)
                    temporary_manifest = os.path.join(tmpdir, f'temp_{filename}')

                    with open(temporary_manifest, 'w', encoding='utf-8') as temp_manifest:
                        transcriptions = []
                        for data_entry in update_data:
                            transcriptions.append(data_entry.get('text', ""))
                            json.dump(data_entry, temp_manifest, ensure_ascii=False)
                            temp_manifest.write('\n')
                if number_of_manifests > 1:
                    temporary_manifest, expanded_audio = handle_multiple_tarr_filepaths(
                        filename, tmpdir, number_of_manifests, expanded_audio[0]
                    )
                else:
                    expanded_audio = expanded_audio[0]

                if self.cfg.train_ds.get("is_tarred", False):
                    hypotheses = self.generate_pseudo_labels(
                        temporary_manifest,
                        tarred_audio_filepaths=expanded_audio,
                        target_transcripts=transcriptions,
                        restore_pc=self.cfg.ipl.restore_pc,
                        batch_size=self.cfg.ipl.batch_size,
                    )

                else:
                    hypotheses = self.generate_pseudo_labels(
                        temporary_manifest,
                        target_transcripts=transcriptions,
                        restore_pc=self.cfg.ipl.restore_pc,
                        batch_size=self.cfg.ipl.batch_size,
                    )

            write_tarr_cache_manifest(
                manifest,
                update_data=shard_manifest_data,
                hypotheses=hypotheses,
                indices=all_indices,
                update_size=update_size,
                use_lhotse=self.cfg.train_ds.get('use_lhotse', False),
            )
            torch.distributed.barrier()

    def combine_cache_hypotheses(self):
        """
        For each dataset combines cache hypotheses from manifests into one final cache manifest.
        Returns:
            final_cache_manifests: List of final cache manifests.
        """
        final_cache_manifests = []
        if self.cfg.train_ds.get("is_tarred", False):

            if not self.cfg.train_ds.get("use_lhotse", False):
                for manifests in self.cfg.ipl.all_cache_manifests:
                    base_path, _ = os.path.split(manifests[0])
                    final_cache = os.path.join(
                        base_path, f'{self.cfg.ipl.cache_prefix}_cache_tarred_audio_manifest.json'
                    )
                    if torch.distributed.get_rank() == 0:
                        create_final_cache_manifest(final_cache, manifests)
                    torch.distributed.barrier()
                    final_cache_manifests.append([final_cache])
            else:
                for i_dataset in self.cfg.ipl.all_cache_manifests:
                    i_dataset = expand_braces(i_dataset)
                    num_manifests = len(i_dataset)
                    base_path, file_name = os.path.split(i_dataset[0])
                    base_file_name = file_name.rsplit('_', 1)[0]
                    dataset_manifests = os.path.join(base_path, f"{base_file_name}_{{{0}..{num_manifests-1}}}.json")
                    final_cache_manifests.append([dataset_manifests])

        return final_cache_manifests

    def generate_pseudo_labels(
        self,
        cache_manifest: Union[List[List[str]], str],
        tarred_audio_filepaths: Union[List[List[str]], str] = None,
        restore_pc: bool = True,
        target_transcripts: List[str] = None,
        batch_size: int = 64,
    ):
        """
        Generates pseudo labels for unlabeled data.
        Args:
            cache_manifest: Temprorary cache file with sampled data.
            tarred_audio_filepaths: Path to tar audio files.
            restore_pc: Whether to restore PC for transcriptions that do not have any.
            target_transcripts: Already existing transcriptions that can be used for restoring PC.c
            batch_size: Batch size used for during inference.
        Returns:
            hypotheses: List of generated labels.
        """
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        self.eval()
        self.encoder.freeze()
        self.decoder.freeze()
        hypotheses = []

        dataloader = self._setup_pseudo_label_dataloader(cache_manifest, tarred_audio_filepaths, batch_size)

        self.preprocessor.featurizer.dither = 0.0
        self.preprocessor.featurizer.pad_to = 0
        sample_idx = 0

        for test_batch in tqdm(dataloader, desc="Transcribing"):
            logits, logits_len, _ = self.forward(
                input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
            )

            logits = logits.cpu()
            if self.cfg.decoding.strategy == "beam":
                best_hyp, all_hyp = self.decoding.ctc_decoder_predictions_tensor(
                    logits,
                    logits_len,
                    return_hypotheses=True,
                )
                if all_hyp:
                    for beams_idx, beams in enumerate(all_hyp):
                        target = target_transcripts[sample_idx + beams_idx]
                        if target and restore_pc:
                            target_split_w = target.split()
                            wer_dist_min = 1000
                            min_pred_text = ""
                            for _, candidate in enumerate(beams):
                                pred_text = candidate.text
                                compare_text = pred_text
                                compare_text = compare_text.lower()
                                compare_text = rm_punctuation(compare_text, ",.?")
                                pred_split_w = compare_text.split()
                                wer_dist = editdistance.eval(target_split_w, pred_split_w)
                                if wer_dist < wer_dist_min:
                                    min_pred_text = pred_text
                                    wer_dist_min = wer_dist
                            hypotheses.append(min_pred_text)
                        else:

                            hypotheses.append(best_hyp[beams_idx].text)
                    sample_idx += logits.shape[0]
                else:
                    hypotheses += [hyp.text for hyp in best_hyp]
            else:
                best_hyp, all_hyp = self.decoding.ctc_decoder_predictions_tensor(
                    logits,
                    logits_len,
                    return_hypotheses=False,
                )
                hypotheses += best_hyp
            del logits
            del logits_len
            del test_batch

        self.train()
        self.preprocessor.featurizer.dither = dither_value
        self.preprocessor.featurizer.pad_to = pad_to_value

        self.encoder.unfreeze()
        self.decoder.unfreeze()
        return hypotheses

    def transcribe(
        self,
        audio: Union[str, List[str], torch.Tensor, np.ndarray],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        If modify this function, please remember update transcribe_partial_audio() in
        nemo/collections/asr/parts/utils/trancribe_utils.py

        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray audio array. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.

        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        return super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            override_config=override_config,
        )

    def change_vocabulary(self, new_vocabulary: List[str], decoding_cfg: Optional[DictConfig] = None):
        """
        Changes vocabulary used during CTC decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        If new_vocabulary == self.decoder.vocabulary then nothing will be changed.

        Args:

            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
            this is target alphabet.

        Returns: None

        """
        if self.decoder.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.decoder.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config['vocabulary'] = new_vocabulary
            new_decoder_config['num_classes'] = len(new_vocabulary)

            del self.decoder
            self.decoder = EncDecCTCModel.from_config_dict(new_decoder_config)
            del self.loss
            self.loss = CTCLoss(
                num_classes=self.decoder.num_classes_with_blank - 1,
                zero_infinity=True,
                reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            )

            if decoding_cfg is None:
                # Assume same decoding config as before
                decoding_cfg = self.cfg.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(CTCDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

            self.decoding = CTCDecoding(
                decoding_cfg=decoding_cfg, vocabulary=OmegaConf.to_container(self.decoder.vocabulary)
            )

            self.wer = WER(
                decoding=self.decoding,
                use_cer=self._cfg.get('use_cer', False),
                dist_sync_on_step=True,
                log_prediction=self._cfg.get("log_prediction", False),
            )

            # Update config
            with open_dict(self.cfg.decoder):
                self._cfg.decoder = new_decoder_config

            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            ds_keys = ['train_ds', 'validation_ds', 'test_ds']
            for key in ds_keys:
                if key in self.cfg:
                    with open_dict(self.cfg[key]):
                        self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

            logging.info(f"Changed decoder to output to {self.decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig):
        """
        Changes decoding strategy used during CTC decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = CTCDecoding(
            decoding_cfg=decoding_cfg, vocabulary=OmegaConf.to_container(self.decoder.vocabulary)
        )

        self.wer = WER(
            decoding=self.decoding,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        self.decoder.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def _setup_dataloader_from_config(self, config: Optional[Dict], do_caching: bool = True):
        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')

        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextBpeDataset(
                    tokenizer=make_parser(
                        labels=config.get('labels', None),
                        name=config.get('parser', 'en'),
                        unk_id=config.get('unk_index', -1),
                        blank_id=config.get('blank_index', -1),
                        do_normalize=config.get('normalize_transcripts', False),
                    ),
                ),
            )

        dataset = audio_to_text_dataset.get_audio_to_text_char_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            preprocessor_cfg=self._cfg.get("preprocessor", None),
            do_caching=do_caching,
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToCharDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        batch_sampler = None
        if config.get('use_semi_sorted_batching', False):
            if not isinstance(dataset, _AudioTextDataset):
                raise RuntimeError(
                    "Semi Sorted Batch sampler can be used with AudioToCharDataset or AudioToBPEDataset "
                    f"but found dataset of type {type(dataset)}"
                )
            # set batch_size and batch_sampler to None to disable automatic batching
            batch_sampler = get_semi_sorted_batch_sampler(self, dataset, config)
            config['batch_size'] = None
            config['drop_last'] = False
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            sampler=batch_sampler,
            batch_sampler=None,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(
        self,
        train_data_config: Optional[Union[DictConfig, Dict]],
        do_caching: bool = True,
        update_limit_train_batches: bool = False,
    ):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config, do_caching=do_caching)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )
            elif update_limit_train_batches:
                # after generation of pseud-labels for tarred datasets.

                self._trainer.limit_train_batches = int(
                    ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_output = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoder_output[0]
        encoded_len = encoder_output[1]
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return (
            log_probs,
            encoded_len,
            greedy_predictions,
        )

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        # Add auxiliary losses, if registered
        loss_value = self.add_auxiliary_losses(loss_value)
        # only computing WER when requested in the logs (same as done for final-layer WER below)
        loss_value, tensorboard_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=((batch_nb + 1) % log_every_n_steps == 0)
        )

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs.update(
            {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )

        if (batch_nb + 1) % log_every_n_steps == 0:
            self.wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self.wer.compute()
            self.wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, sample_id = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        transcribed_texts, _ = self.wer.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs,
            decoder_lengths=encoded_len,
            return_hypotheses=False,
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, transcribed_texts))

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        loss_value, metrics = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            log_wer_num_denom=True,
            log_prefix="val_",
        )

        self.wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        self.wer.reset()
        metrics.update({'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom, 'val_wer': wer})

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        metrics = super().multi_validation_epoch_end(outputs, dataloader_idx)
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")
        return metrics

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        metrics = super().multi_test_epoch_end(outputs, dataloader_idx)
        self.finalize_interctc_metrics(metrics, outputs, prefix="test_")
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def _setup_pseudo_label_dataloader(self, cache_manifest: str, audio_tar: str = None, batch_size: int = 64):

        if self.cfg.train_ds.get("is_tarred", False):

            dl_config = {
                'manifest_filepath': cache_manifest,
                'tarred_audio_filepaths': audio_tar,
                'sample_rate': self.preprocessor._sample_rate,
                'labels': OmegaConf.to_container(self.decoder.vocabulary),
                'is_tarred': True,
                'use_lhotse': True,
                'shard_manifests': False,
                'tarred_shard_strategy': 'replicate',
                'batch_size': batch_size,
                'drop_last': False,
                'trim_silence': False,
                'shuffle': False,
                'shuffle_n': 0,
                'num_workers': self.cfg.train_ds.num_workers,
                'pin_memory': True,
                'tarred_random_access': True,
            }

            dl_config = OmegaConf.create(dl_config)

            return get_lhotse_dataloader_from_config(
                dl_config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextBpeDataset(
                    tokenizer=make_parser(labels=self.decoder.vocabulary, do_normalize=False)
                ),
            )
        else:

            dl_config = {
                'manifest_filepath': cache_manifest,
                'sample_rate': self.preprocessor._sample_rate,
                'labels': self.joint.vocabulary,
                'batch_size': batch_size,
                'trim_silence': False,
                'shuffle': False,
                'num_workers': self.cfg.train_ds.num_workers,
                'pin_memory': True,
            }

        dataset = audio_to_text_dataset.get_char_dataset(config=dl_config, augmentor=None, do_caching=False)
        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False,
            num_workers=self.cfg.train_ds.num_workers,
            pin_memory=True,
        )

    """ Transcription related methods """

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        super()._transcribe_on_begin(audio, trcfg)

        # Freeze the encoder and decoure_exder modules
        self.encoder.freeze()
        self.decoder.freeze()

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        super()._transcribe_on_end(trcfg)

        # Unfreeze the encoder and decoder modules
        self.encoder.unfreeze()
        self.decoder.unfreeze()

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        logits, logits_len, greedy_predictions = self.forward(input_signal=batch[0], input_signal_length=batch[1])
        output = dict(logits=logits, logits_len=logits_len)
        del greedy_predictions
        return output

    def _transcribe_output_processing(self, outputs, trcfg: TranscribeConfig) -> GenericTranscriptionType:
        logits = outputs.pop('logits')
        logits_len = outputs.pop('logits_len')

        current_hypotheses, all_hyp = self.decoding.ctc_decoder_predictions_tensor(
            logits,
            decoder_lengths=logits_len,
            return_hypotheses=trcfg.return_hypotheses,
        )
        if trcfg.return_hypotheses:
            if logits.is_cuda:
                # See comment in
                # ctc_greedy_decoding.py::GreedyCTCInfer::forward() to
                # understand this idiom.
                logits_cpu = torch.empty(logits.shape, dtype=logits.dtype, device=torch.device("cpu"), pin_memory=True)
                logits_cpu.copy_(logits, non_blocking=True)
            else:
                logits_cpu = logits
            logits_len = logits_len.cpu()
            # dump log probs per file
            for idx in range(logits_cpu.shape[0]):
                current_hypotheses[idx].y_sequence = logits_cpu[idx][: logits_len[idx]]
                if current_hypotheses[idx].alignments is None:
                    current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence
            del logits_cpu

        # cleanup memory
        del logits, logits_len

        hypotheses = []
        if all_hyp is None:
            hypotheses += current_hypotheses
        else:
            hypotheses += all_hyp

        return hypotheses

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': OmegaConf.to_container(self.decoder.vocabulary),
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'channel_selector': config.get('channel_selector', None),
        }
        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5Base-En",
            description="QuartzNet15x5 model trained on six datasets: LibriSpeech, Mozilla Common Voice (validated clips from en_1488h_2019-12-10), WSJ, Fisher, Switchboard, and NSC Singapore English. It was trained with Apex/Amp optimization level O1 for 600 epochs. The model achieves a WER of 3.79% on LibriSpeech dev-clean, and a WER of 10.05% on dev-other. Please visit https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels for further details.",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_quartznet15x5/versions/1.0.0rc1/files/stt_en_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_jasper10x5dr",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_jasper10x5dr",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_jasper10x5dr/versions/1.0.0rc1/files/stt_en_jasper10x5dr.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ca_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ca_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ca_quartznet15x5/versions/1.0.0rc1/files/stt_ca_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_it_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_it_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_quartznet15x5/versions/1.0.0rc1/files/stt_it_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_fr_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_quartznet15x5/versions/1.0.0rc1/files/stt_fr_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_quartznet15x5/versions/1.0.0rc1/files/stt_es_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_quartznet15x5/versions/1.0.0rc1/files/stt_de_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_pl_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_pl_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_pl_quartznet15x5/versions/1.0.0rc1/files/stt_pl_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ru_quartznet15x5",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ru_quartznet15x5",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_quartznet15x5/versions/1.0.0rc1/files/stt_ru_quartznet15x5.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_citrinet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_zh_citrinet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_citrinet_512/versions/1.0.0rc1/files/stt_zh_citrinet_512.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_zh_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_citrinet_1024_gamma_0_25/versions/1.0.0/files/stt_zh_citrinet_1024_gamma_0_25.nemo",
        )

        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_zh_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_citrinet_1024_gamma_0_25/versions/1.0.0/files/stt_zh_citrinet_1024_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="asr_talknet_aligner",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:asr_talknet_aligner",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/asr_talknet_aligner/versions/1.0.0rc1/files/qn5x5_libri_tts_phonemes.nemo",
        )
        results.append(model)

        return results

    @property
    def wer(self):
        return self._wer

    @wer.setter
    def wer(self, wer):
        self._wer = wer
