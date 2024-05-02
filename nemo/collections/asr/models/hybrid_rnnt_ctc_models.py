# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, List, Optional, Tuple
import editdistance

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm
import tempfile
from nemo.collections.asr.data import audio_to_text_dataset

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, InterCTCMixin, TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import TranscriptionReturnType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.collections.common.parts.preprocessing.parsers import make_parser
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging, model_utils
from nemo.collections.asr.parts.utils.ipl_utils import *
from nemo.collections.asr.data.audio_to_text import cache_datastore_manifests, expand_sharded_filepaths
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config


class EncDecHybridRNNTCTCModel(EncDecRNNTModel, ASRBPEMixin, InterCTCMixin):
    """Base class for hybrid RNNT/CTC models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        if cfg.get("ipl", None):
            with open_dict(cfg.ipl):
                
                cfg.ipl.num_all_files, cfg.ipl.num_cache_files = count_files_for_pseudo_labeling(
                    cfg.ipl.manifest_filepath, cfg.ipl.get('dataset_weights', None)
                )
                if not cfg.train_ds.get("is_tarred", False):
                    if not cfg.ipl.get("cache_manifest", None):
                        cfg.ipl.cache_manifest = str(Path.cwd() / f"{cfg.ipl.cache_prefix}_pseudo_labeled.json")
                else:
                    cfg.ipl.cache_manifest = []
        super().__init__(cfg=cfg, trainer=trainer)

        if 'aux_ctc' not in self.cfg:
            raise ValueError(
                "The config need to have a section for the CTC decoder named as aux_ctc for Hybrid models."
            )
        with open_dict(self.cfg.aux_ctc):
            if "feat_in" not in self.cfg.aux_ctc.decoder or (
                not self.cfg.aux_ctc.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self.cfg.aux_ctc.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self.cfg.aux_ctc.decoder or not self.cfg.aux_ctc.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.aux_ctc.decoder.num_classes < 1 and self.cfg.aux_ctc.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.aux_ctc.decoder.num_classes, len(self.cfg.aux_ctc.decoder.vocabulary)
                    )
                )
                self.cfg.aux_ctc.decoder["num_classes"] = len(self.cfg.aux_ctc.decoder.vocabulary)

        self.ctc_decoder = EncDecRNNTModel.from_config_dict(self.cfg.aux_ctc.decoder)
        self.ctc_loss_weight = self.cfg.aux_ctc.get("ctc_loss_weight", 0.5)

        self.ctc_loss = CTCLoss(
            num_classes=self.ctc_decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
        )

        ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
        if ctc_decoding_cfg is None:
            ctc_decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg

        self.ctc_decoding = CTCDecoding(self.cfg.aux_ctc.decoding, vocabulary=self.ctc_decoder.vocabulary)
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )

        # setting the RNNT decoder as the default one
        self.cur_decoder = "rnnt"

        # setting up interCTC loss (from InterCTCMixin)
        self.setup_interctc(decoder_name='ctc_decoder', loss_name='ctc_loss', wer_name='ctc_wer')

    def on_fit_start(self):
        """
            Cache datastore manifests for non tarred unlabeled data 
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
            
            self.build_cache(self.cfg.ipl.manifest_filepath, self.cfg.ipl.get("tarred_audio_filepaths", None), update_whole_cache=True)

            self.encoder.set_dropout(self.cfg.ipl.dropout)            
            self.cfg.ipl.m_epochs -= 1
            needs_update = False

        if self.cfg.ipl.m_epochs == -1 and self.cfg.ipl.n_l_epochs > 0:
            self.cfg.ipl.n_l_epochs -= 1
        else: 
            if needs_update:
 
                self.build_cache(self.cfg.ipl.manifest_filepath, self.cfg.ipl.get("tarred_audio_filepaths", None), update_whole_cache=False)
                final_cache_manifests = self.combine_cache_hypotheses()    
            else:
        
                final_cache_manifests = self.combine_cache_hypotheses()    
                if self.cfg.ipl.n_l_epochs == 0:
                    if self.cfg.train_ds.get("is_tarred", False):
                        if not self.cfg.train_ds.get("use_lhotse", False):
                            if isinstance(self.cfg.ipl.tarred_audio_filepaths, str):
                                if isinstance(self.cfg.train_ds.tarred_audio_filepaths, str):
                                    self.cfg.train_ds.tarred_audio_filepaths = [[self.cfg.train_ds.tarred_audio_filepaths],[self.cfg.ipl.tarred_audio_filepaths] ]
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

                    self.cfg.ipl.n_l_epochs -= 1
                    self.trainer.reload_dataloaders_every_n_epochs = 1

            torch.distributed.barrier()

            self.setup_training_data(self.cfg.train_ds, do_caching = False, update_limit_train_batches=True)
    
    
    def build_cache(self, manifests: Union[List[List[str]], str], tarred_audio_filepaths: Union[List[List[str]], str] = None, update_whole_cache = True):
        """
        Function to build cache file for maintaining pseudo labels
        Args:
            update_whole_cache: (bool) Indicates whether to update the entire cache or only a portion of it based on sampling.
            manifests:  Manifest file(s) from which pseudo labels will be generated
            tarred_audio_filepaths: Tar file paths for tarred datasets
                
        """
        if self.cfg.train_ds.get("is_tarred", False):
            if isinstance(manifests, str):
                manifests = [[manifests]]
            if isinstance(tarred_audio_filepaths, str):
                tarred_audio_filepaths = [[tarred_audio_filepaths]]
            if update_whole_cache:
                self.create_tar_cache_hypotheses(manifests, tarred_audio_filepaths)
            else:
                self.update_tarr_cache_hyppotheses(tarred_audio_filepaths)
        else:
            self.create_cache_hypotheses(manifests, update_whole_cache)
    

    def create_cache_hypotheses(self, manifests: Union[List[List[str]], str], update_whole_cache: bool = True):
        """
        Function to create cache file for unlabeled dataset
        Args:
            update_whole_cache: Indicates whether to update the entire cache or only a portion of it based on sampling.
            manifests:  Manifest file(s) from which pseudo labels will be generated
        """

        whole_pseudo_data = []
        update_data = []

        manifest_paths =  [manifests] if isinstance(manifests, str) else manifests
        dataset_weights = self.cfg.ipl.get("dataset_weights", [1] * len(manifest_paths))

        if not isinstance(dataset_weights, ListConfig) and not isinstance(dataset_weights, List) :
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

            hypotheses = self.generate_pseudo_labels(temporary_manifest,
                                                    target_transcripts=transcriptions, 
                                                    restore_pc=self.cfg.ipl.restore_pc,
                                                    )
        torch.distributed.barrier()
        gathered_hypotheses = [None]  * torch.distributed.get_world_size()
        gathered_data = [None]  * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_data, update_data)
        torch.distributed.all_gather_object(gathered_hypotheses, hypotheses)
        if torch.distributed.get_rank() == 0:
            write_cache_manifest(self.cfg.ipl.cache_manifest, gathered_hypotheses, gathered_data, update_whole_cache)
        torch.distributed.barrier()


    def create_tar_cache_hypotheses(self, manifests: Union[List[List[str]], str],  tarred_audio_filepaths: Union[List[List[str]], str]):
        """
        Function to create cache file for tarred unlabeled dataset for the first time
        Args:
            manifests:  Manifest file(s) from which pseudo labels will be generated
            tarred_audio_filepaths: Tar file paths for tarred datasets
        """

        self.cfg.ipl.cache_manifest = []
        for manifest, tarred_audio_filepath in zip(manifests, tarred_audio_filepaths):
            with tempfile.TemporaryDirectory() as tmpdir:
                
                expanded_audio = expand_sharded_filepaths(tarred_audio_filepath[0], shard_strategy = 'scatter', world_size=self.world_size, global_rank=self.global_rank)
                expand_manifests = expand_sharded_filepaths(manifest[0], shard_strategy = 'scatter', world_size=self.world_size, global_rank=self.global_rank)
                number_of_manifests = len(expand_manifests)
            
                shard_manifest_data = []
                cache_manifest = []
                transcriptions = []
                for _, manifest_path in enumerate(expand_manifests): 
                
                    manifest_data = process_manifest(manifest_path)      
                    shard_manifest_data.append(manifest_data)

                    base_path, filename = os.path.split(manifest_path)
                    cache_file = os.path.join(base_path, f'{self.cfg.ipl.cache_prefix}_cache_{filename}')
                    cache_manifest.append(cache_file)
                    
                    temporary_manifest=os.path.join(tmpdir, f'temp_{filename}')
                    
                    with open(temporary_manifest, 'w', encoding='utf-8') as temp_manifest:
                        
                        for data_entry in manifest_data:
                            
                            if not data_entry.get("text", None):
                                data_entry['text'] = ""
                            transcriptions.append(data_entry.get('text', ""))
                            json.dump(data_entry, temp_manifest, ensure_ascii=False)
                            temp_manifest.write('\n')

                if number_of_manifests > 1:
                    temporary_manifest, expanded_audio = handle_multiple_tarr_filepaths(filename, tmpdir, number_of_manifests, expanded_audio[0])
                else:
                    expanded_audio = expanded_audio[0]

                if self.cfg.train_ds.get("is_tarred", False):

                    hypotheses = self.generate_pseudo_labels(cache_manifest = temporary_manifest,
                                                        tarred_audio_filepaths=expanded_audio, 
                                                        target_transcripts=None,
                                                        restore_pc=self.cfg.ipl.restore_pc,
                                                        )
                
                else:
                    hypotheses = self.generate_pseudo_labels(manifest,
                                                            target_transcripts=None,
                                                                restore_pc=self.cfg.ipl.restore_pc,
                                                                    )
            
                write_tarr_cache_manifest(cache_manifest, update_data=shard_manifest_data, hypotheses=hypotheses)
                self.cfg.ipl.cache_manifest.append(cache_manifest)


    def update_tarr_cache_hyppotheses(self, tarred_audio_filepaths: Union[List[List[str]], str]):
        """
            With given probability randomly chooses part of the cache hypotheses, generates new pseudo labels for them and updates the cache.
            Args:
                tarred_audio_filepaths: Path to tarred audio files.

        """
        for manifest, tarred_audio_filepath in zip(self.cfg.ipl.cache_manifest, tarred_audio_filepaths):
            with tempfile.TemporaryDirectory() as tmpdir:
                expanded_audio = expand_sharded_filepaths(tarred_audio_filepath[0], shard_strategy = 'scatter', world_size=self.world_size, global_rank=self.global_rank)
                shard_manifest_data = []
                number_of_manifests = len(manifest)
                for _, manifest_path in enumerate(manifest): 
                
                    manifest_data = process_manifest(manifest_path)  
                    
                    random.shuffle(manifest_data)
                    shard_manifest_data.append(manifest_data)  
            
                    update_size = int(len(manifest_data) * self.cfg.ipl.p_cache)
                    update_data = manifest_data[:update_size]

                    _, filename = os.path.split(manifest_path)
                    temporary_manifest = os.path.join(tmpdir, f'temp_{filename}')
            
                    with open(temporary_manifest, 'w', encoding='utf-8') as temp_manifest:
                        transcriptions = []
                        for data_entry in update_data:
                            transcriptions.append(data_entry.get('text', ""))
                            json.dump(data_entry, temp_manifest, ensure_ascii=False)
                            temp_manifest.write('\n')
                if number_of_manifests > 1:
                    temporary_manifest, expanded_audio = handle_multiple_tarr_filepaths(filename, tmpdir, number_of_manifests, expanded_audio[0])
                else:
                    expanded_audio = expanded_audio[0]
    

                if self.cfg.train_ds.get("is_tarred", False):
                    hypotheses = self.generate_pseudo_labels(temporary_manifest,
                                                        tarred_audio_filepaths=expanded_audio, 
                                                        target_transcripts=transcriptions,
                                                        restore_pc=self.cfg.ipl.restore_pc,
                                                        )
                
                else:
                    hypotheses = self.generate_pseudo_labels(temporary_manifest,
                                                            target_transcripts=transcriptions,
                                                            restore_pc=self.cfg.ipl.restore_pc,
                                                            )
                torch.distributed.barrier()
    
            write_tarr_cache_manifest(manifest, update_data=shard_manifest_data, hypotheses=hypotheses, update_size=update_size)


    def combine_cache_hypotheses(self):
        """
        For each dataset combines cache hypotheses from manifests into one final cache manifest
        Returns:
            final_cache_manifests: List of final cache manifests
        """
        if self.cfg.train_ds.get("is_tarred", False):
            torch.distributed.barrier()
            all_cache_gathered = [None]  * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(all_cache_gathered, self.cfg.ipl.cache_manifest) 
            result = [[item for sublist in group for item in sublist] for group in zip(*all_cache_gathered)]

            final_cache_manifests = []
            if not self.cfg.train_ds.get("use_lhotse", False):
                for manifests in result:
                    base_path, _ = os.path.split(manifests[0])
                    final_cache = os.path.join(base_path, f'{self.cfg.ipl.cache_prefix}_cache_tarred_audio_manifest.json')
                    if torch.distributed.get_rank() == 0:
                        create_final_cache_manifest(final_cache, manifests)

                    final_cache_manifests.append([final_cache])
            torch.distributed.barrier() 
        return final_cache_manifests
        

    def generate_pseudo_labels(
        self,
        cache_manifest: Union[List[List[str]], str],
        tarred_audio_filepaths: Union[List[List[str]], str] = None,
        restore_pc: bool = True,
        target_transcripts: List[str] = None):
        """
        Generates pseudo labels for unlabeled data.
        Args:
            cache_manifest: Temprorary cache file with sampled data.
            tarred_audio_filepaths: path to tarr audio files
            batch_size: Batch size used for during inference.
            restore_pc: Whether to restore PC for transcriptions that do not have any.
            target_transcripts: Already existing transcriptions that can be used for restoring PC
        Returns:
            target_transcripts: List of generated labels.
        """
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        self.eval()
        self.encoder.freeze()
        self.decoder.freeze()
        self.joint.freeze()
        self.ctc_decoder.freeze()
        hypotheses = []


        dataloader = self._setup_pseudo_label_dataloader(cache_manifest, tarred_audio_filepaths, self.cfg.ipl.batch_size)
        self.preprocessor.featurizer.dither = 0.0
        self.preprocessor.featurizer.pad_to = 0
        sample_idx = 0
        count = 0
        for test_batch in tqdm(dataloader, desc="Transcribing"):
            count += 1
            encoded, encoded_len = self.forward(
                input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
            )

            logits = self.ctc_decoder(encoder_output=encoded)
            logits = logits.cpu()
            if self.cfg.aux_ctc.decoding.strategy == "beam":
                
                best_hyp, all_hyp = self.ctc_decoding.ctc_decoder_predictions_tensor(
                logits, encoded_len, return_hypotheses=True,
                )
                if all_hyp:
                    for beams_idx, beams in enumerate(all_hyp):
                        if restore_pc:
                            target = target_transcripts[sample_idx + beams_idx ]
                            if target != "":
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
                                        wer_dist_min =  wer_dist
                                hypotheses.append(min_pred_text)
                            else:
                                hypotheses.append(best_hyp[beams_idx].text)
                        else:
                            hypotheses.append(best_hyp[beams_idx].text)
                    sample_idx += logits.shape[0]
                else:
                    hypotheses += [hyp.text for hyp in best_hyp]
            else:
                best_hyp, all_hyp = self.ctc_decoding.ctc_decoder_predictions_tensor(
                logits, encoded_len, return_hypotheses=False,)
                hypotheses += best_hyp
            
            del logits
            del encoded
            del test_batch
        
    
        self.train()
        self.preprocessor.featurizer.dither = dither_value
        self.preprocessor.featurizer.pad_to = pad_to_value

        self.encoder.unfreeze()
        self.decoder.unfreeze()
        self.joint.unfreeze()
  
        self.ctc_decoder.unfreeze()
        return hypotheses
    
    def _setup_pseudo_label_dataloader(self, manifest_filepaths: Union[List[List[str]], str], tarred_audio_filepaths: Union[List[List[str]], str] = None, batch_size: int = 64):
        
        """
        Setup function for a data loader for unlabeled dataset

        Args:
            manifest_filepaths: Manifests containing information of unlabeled dataset. For tarred dataset manifests should be sharded
            tarred_audio_filepaths: Tarr audio files which should correspond to manifest files.
            batch_size: batch size to use during inference.
                
        Returns:
            A DataLoader for the given audio file(s).
        """

        if self.cfg.train_ds.get("is_tarred", False):

            dl_config = {
                'manifest_filepath': manifest_filepaths,
                'tarred_audio_filepaths': tarred_audio_filepaths,
                'sample_rate': self.preprocessor._sample_rate,
                'labels': self.joint.vocabulary,
                'is_tarred': True,
                'use_lhotse' : True,
                'shard_manifests': False,
                'tarred_shard_strategy': 'replicate',
                'batch_size': batch_size,
                'drop_last': False,
                'trim_silence': False,
                'shuffle': False,
                'shuffle_n': 0,
                'num_workers': self.cfg.train_ds.num_workers,
                'pin_memory': True,
                'random_access': True,
                }
            
            dl_config = OmegaConf.create(dl_config)

            return get_lhotse_dataloader_from_config(
                dl_config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextBpeDataset(tokenizer=make_parser(
                        labels=self.joint.vocabulary,
                        do_normalize=False)),
                pseudo_label_gen=True)
        else:

            dl_config = {
                'manifest_filepath': manifest_filepaths,
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
            batch_size=dl_config['batch_size'],
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False,
            num_workers=dl_config['num_workers'],
            pin_memory=True,
        )
    
    
    @torch.no_grad()
    def transcribe(
        self,
        audio: List[str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:

            audio: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            logprobs: (bool) whether to return ctc logits insted of hypotheses

        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """
        if self.cur_decoder not in ["ctc", "rnnt"]:
            raise ValueError(
                f"{self.cur_decoder} is not supported for cur_decoder. Supported values are ['ctc', 'rnnt']"
            )

        return super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            partial_hypothesis=partial_hypothesis,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            override_config=override_config,
        )

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        super()._transcribe_on_begin(audio, trcfg)

        if hasattr(self, 'ctc_decoder'):
            self.ctc_decoder.freeze()

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        super()._transcribe_on_end(trcfg)

        if hasattr(self, 'ctc_decoder'):
            self.ctc_decoder.unfreeze()

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        if self.cur_decoder == "rnnt":
            return super()._transcribe_forward(batch, trcfg)

        # CTC Path
        encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])

        logits = self.ctc_decoder(encoder_output=encoded)
        output = dict(logits=logits, encoded_len=encoded_len)

        del encoded
        return output

    def _transcribe_output_processing(
        self, outputs, trcfg: TranscribeConfig
    ) -> Tuple[List['Hypothesis'], List['Hypothesis']]:
        if self.cur_decoder == "rnnt":
            return super()._transcribe_output_processing(outputs, trcfg)

        # CTC Path
        logits = outputs.pop('logits')
        encoded_len = outputs.pop('encoded_len')

        best_hyp, all_hyp = self.ctc_decoding.ctc_decoder_predictions_tensor(
            logits, encoded_len, return_hypotheses=trcfg.return_hypotheses,
        )
        logits = logits.cpu()

        if trcfg.return_hypotheses:
            # dump log probs per file
            for idx in range(logits.shape[0]):
                best_hyp[idx].y_sequence = logits[idx][: encoded_len[idx]]
                if best_hyp[idx].alignments is None:
                    best_hyp[idx].alignments = best_hyp[idx].y_sequence

        # DEPRECATED?
        # if logprobs:
        #     for logit, elen in zip(logits, encoded_len):
        #         logits_list.append(logit[:elen])

        del logits, encoded_len

        hypotheses = []
        all_hypotheses = []

        hypotheses += best_hyp
        if all_hyp is not None:
            all_hypotheses += all_hyp
        else:
            all_hypotheses += best_hyp

        return (hypotheses, all_hypotheses)

    def change_vocabulary(
        self,
        new_vocabulary: List[str],
        decoding_cfg: Optional[DictConfig] = None,
        ctc_decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning a pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
                this is target alphabet.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            ctc_decoding_cfg: A config for CTC decoding, which is optional and can be used to change decoding type.

        Returns: None

        """
        super().change_vocabulary(new_vocabulary=new_vocabulary, decoding_cfg=decoding_cfg)

        # set up the new tokenizer for the CTC decoder
        if hasattr(self, 'ctc_decoder'):
            if self.ctc_decoder.vocabulary == new_vocabulary:
                logging.warning(
                    f"Old {self.ctc_decoder.vocabulary} and new {new_vocabulary} match. Not changing anything."
                )
            else:
                if new_vocabulary is None or len(new_vocabulary) == 0:
                    raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
                decoder_config = self.ctc_decoder.to_config_dict()
                new_decoder_config = copy.deepcopy(decoder_config)
                new_decoder_config['vocabulary'] = new_vocabulary
                new_decoder_config['num_classes'] = len(new_vocabulary)

                del self.ctc_decoder
                self.ctc_decoder = EncDecHybridRNNTCTCModel.from_config_dict(new_decoder_config)
                del self.ctc_loss
                self.ctc_loss = CTCLoss(
                    num_classes=self.ctc_decoder.num_classes_with_blank - 1,
                    zero_infinity=True,
                    reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
                )

                if ctc_decoding_cfg is None:
                    # Assume same decoding config as before
                    logging.info("No `ctc_decoding_cfg` passed when changing decoding strategy, using internal config")
                    ctc_decoding_cfg = self.cfg.aux_ctc.decoding

                # Assert the decoding config with all hyper parameters
                ctc_decoding_cls = OmegaConf.structured(CTCDecodingConfig)
                ctc_decoding_cls = OmegaConf.create(OmegaConf.to_container(ctc_decoding_cls))
                ctc_decoding_cfg = OmegaConf.merge(ctc_decoding_cls, ctc_decoding_cfg)

                self.ctc_decoding = CTCDecoding(decoding_cfg=ctc_decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

                self.ctc_wer = WER(
                    decoding=self.ctc_decoding,
                    use_cer=self.ctc_wer.use_cer,
                    log_prediction=self.ctc_wer.log_prediction,
                    dist_sync_on_step=True,
                )

                # Update config
                with open_dict(self.cfg.aux_ctc):
                    self.cfg.aux_ctc.decoding = ctc_decoding_cfg

                with open_dict(self.cfg.aux_ctc):
                    self.cfg.aux_ctc.decoder = new_decoder_config

                ds_keys = ['train_ds', 'validation_ds', 'test_ds']
                for key in ds_keys:
                    if key in self.cfg:
                        with open_dict(self.cfg[key]):
                            self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

                logging.info(f"Changed the tokenizer of the CTC decoder to {self.ctc_decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig = None, decoder_type: str = None):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            decoder_type: (str) Can be set to 'rnnt' or 'ctc' to switch between appropriate decoder in a
                model having RNN-T and CTC decoders. Defaults to None, in which case RNN-T decoder is
                used. If set to 'ctc', it raises error if 'ctc_decoder' is not an attribute of the model.
        """
        if decoder_type is None or decoder_type == 'rnnt':
            self.cur_decoder = "rnnt"
            return super().change_decoding_strategy(decoding_cfg=decoding_cfg)

        assert decoder_type == 'ctc' and hasattr(self, 'ctc_decoder')
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.aux_ctc.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.ctc_decoding = CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.ctc_wer.use_cer,
            log_prediction=self.ctc_wer.log_prediction,
            dist_sync_on_step=True,
        )

        self.ctc_decoder.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.aux_ctc):
            self.cfg.aux_ctc.decoding = decoding_cfg

        self.cur_decoder = "ctc"
        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.aux_ctc.decoding)}")

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        if (sample_id + 1) % log_every_n_steps == 0:
            compute_wer = True
        else:
            compute_wer = False

        # If fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:  # If fused Joint-Loss-WER is used
            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_rnnt_loss'] = loss_value
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

        # note that we want to apply interctc independent of whether main ctc
        # loss is used or not (to allow rnnt + interctc training).
        # assuming ``ctc_loss_weight=0.3`` and interctc is applied to a single
        # layer with weight of ``0.1``, the total loss will be
        # ``loss = 0.9 * (0.3 * ctc_loss + 0.7 * rnnt_loss) + 0.1 * interctc_loss``
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )
        tensorboard_logs.update(additional_logs)
        tensorboard_logs['train_loss'] = loss_value
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO: add support for CTC decoding
        signal, signal_len, transcript, transcript_len, sample_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_pass(self, batch, batch_idx, dataloader_idx):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}
        loss_value = None

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )
                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )
            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        log_probs = self.ctc_decoder(encoder_output=encoded)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value
        self.ctc_wer.update(
            predictions=log_probs, targets=transcript, targets_lengths=transcript_len, predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            # overriding total loss value. Note that the previous
            # rnnt + ctc loss is available in metrics as "val_final_loss" now
            tensorboard_logs['val_loss'] = loss_value
        tensorboard_logs.update(additional_logs)
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return tensorboard_logs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        tensorboard_logs = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(tensorboard_logs)
        else:
            self.validation_step_outputs.append(tensorboard_logs)

        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}
        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['val_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['val_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['val_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom
        metrics = {**val_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")
        return metrics

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss_log = {'test_loss': test_loss_mean}
        else:
            test_loss_log = {}
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**test_loss_log, 'test_wer': wer_num.float() / wer_denom}

        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['test_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['test_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['test_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        metrics = {**test_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="test_")
        return metrics

    # EncDecRNNTModel is exported in 2 parts
    def list_export_subnets(self):
        if self.cur_decoder == 'rnnt':
            return ['encoder', 'decoder_joint']
        else:
            return ['self']

    @property
    def output_module(self):
        if self.cur_decoder == 'rnnt':
            return self.decoder
        else:
            return self.ctc_decoder

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results
