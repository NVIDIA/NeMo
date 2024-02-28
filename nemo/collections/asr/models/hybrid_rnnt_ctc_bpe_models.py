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
import os
from typing import Dict, List, Optional, Union

import torch
import json
import tempfile
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
import editdistance
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTBPEDecodingConfig
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils
from nemo.collections.asr.parts.utils.ipl_utils import *


class EncDecHybridRNNTCTCBPEModel(EncDecHybridRNNTCTCModel, ASRBPEMixin):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss and subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Tokenizer is necessary for this model
        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        with open_dict(cfg):
            cfg.labels = ListConfig(list(vocabulary))

        with open_dict(cfg.decoder):
            cfg.decoder.vocab_size = len(vocabulary)

        with open_dict(cfg.joint):
            cfg.joint.num_classes = len(vocabulary)
            cfg.joint.vocabulary = ListConfig(list(vocabulary))
            cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
            cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

        # setup auxiliary CTC decoder
        if 'aux_ctc' not in cfg:
            raise ValueError(
                "The config need to have a section for the CTC decoder named as aux_ctc for Hybrid models."
            )

        with open_dict(cfg):
            if self.tokenizer_type == "agg":
                cfg.aux_ctc.decoder.vocabulary = ListConfig(vocabulary)
            else:
                cfg.aux_ctc.decoder.vocabulary = ListConfig(list(vocabulary.keys()))

        if cfg.aux_ctc.decoder["num_classes"] < 1:
            logging.info(
                "\nReplacing placholder number of classes ({}) with actual number of classes - {}".format(
                    cfg.aux_ctc.decoder["num_classes"], len(vocabulary)
                )
            )
            cfg.aux_ctc.decoder["num_classes"] = len(vocabulary)
        
        if cfg.get("ipl", None):
            with open_dict(cfg.ipl):
                cfg.ipl.num_all_files, cfg.ipl.num_cache_files = count_files_for_pseudo_labeling(cfg.ipl.manifest_filepath,
                                                                                        cfg.ipl.get('dataset_weights', None))
                if not cfg.ipl.get("cache_manifest", None):
                    cfg.ipl.cache_manifest = "/tmp/manifest_pseudo_labeled.json"

        super().__init__(cfg=cfg, trainer=trainer)

        # Setup decoding object
        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
        )

        # Setup wer object
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self.cfg.get('use_cer', False),
            log_prediction=self.cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Setup CTC decoding
        ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
        if ctc_decoding_cfg is None:
            ctc_decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg
        self.ctc_decoding = CTCBPEDecoding(self.cfg.aux_ctc.decoding, tokenizer=self.tokenizer)

        # Setup CTC WER
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )

        # setting the RNNT decoder as the default one
        self.cur_decoder = "rnnt"

    def _setup_dataloader_from_config(self, config: Optional[Dict]):

        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextBpeDataset(tokenizer=self.tokenizer,),
            )

        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToBPEDALIDataset):
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

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def on_train_epoch_end(self):
        
        """
        This function is mainly used for iterative pseudo labeling algorithm.
        To make it work in config file 'ipl' parameters should be provided.

        """
        if not self.cfg.get("ipl"):
            return

        if self.cfg.ipl.m_updates > 0:
            self.cfg.ipl.m_updates -= 1
            return
        needs_update = True
        if self.cfg.ipl.m_updates == 0:
            data, hypotheses = self.update_cache_hypotheses()
            torch.distributed.barrier()

            gathered_hypotheses = [None]  * torch.distributed.get_world_size()
            gathered_data = [None]  * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_data, data)
            torch.distributed.all_gather_object(gathered_hypotheses, hypotheses)

            if torch.distributed.get_rank() == 0:
                write_cache_manifest(self.cfg.ipl.cache_manifest, gathered_hypotheses, gathered_data)
            torch.distributed.barrier()

            self.encoder.set_dropout(self.cfg.ipl.dropout)            
            self.cfg.ipl.m_updates -= 1
            needs_update = False
        if self.cfg.ipl.m_updates == -1 and self.cfg.ipl.n_l_updates > 0:
            self.cfg.ipl.n_l_updates -= 1
        else: 
            if needs_update:
                data, hypotheses = self.update_cache_hypotheses(False)
                torch.distributed.barrier()
                gathered_hypotheses = [None] * torch.distributed.get_world_size()
                all_random_samples = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(all_random_samples, data)
                torch.distributed.all_gather_object(gathered_hypotheses, hypotheses)

                if torch.distributed.get_rank() == 0:
                    write_cache_manifest(self.cfg.ipl.cache_manifest, gathered_hypotheses, all_random_samples, False)
                torch.distributed.barrier()
            
            if self.cfg.ipl.n_l_updates == 0:
                if isinstance(self.cfg.train_ds.manifest_filepath, str):
                    self.cfg.train_ds.manifest_filepath = [self.cfg.train_ds.manifest_filepath]
                    self.cfg.train_ds.manifest_filepath.append(self.cfg.ipl.cache_manifest)
                else:
                    self.cfg.train_ds.manifest_filepath.append(self.cfg.ipl.cache_manifest)
           
                self.cfg.ipl.n_l_updates -= 1
                self.trainer.reload_dataloaders_every_n_epochs = 1
   
            self.setup_training_data(self.cfg.train_ds)

    def update_cache_hypotheses(self, update_whole_cache=True):
        """
        Gathers data for updating cache file for pseudo labeling.
        Args:
            update_whole_cache: (bool) Indicates whether to update the entire cache or only a portion of it based on sampling.

        Returns:
            update_data: (list) The sampled data entries from the manifest files that will be used to update the cache.
            hypotheses: (list) The generated pseudo labels corresponding to the `update_data`.

        """

        whole_pseudo_data = []
        update_data = []

        manifest_paths =  [self.cfg.ipl.manifest_filepath] if isinstance(self.cfg.ipl.manifest_filepath, str) else self.cfg.ipl.manifest_filepath 
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
                                                    batch_size = self.cfg.train_ds.batch_size,
                                                    num_workers=self.cfg.train_ds.num_workers,
                                                    )
            return update_data, hypotheses
    
    def generate_pseudo_labels(
        self,
        cache_manifest: str,
        batch_size: int = 4,
        num_workers: int = 4,
        restore_pc: bool = True,
        target_transcripts: List[str] = None):
        """
        Generates pseudo labels for unlabeled data.
        Args:
            cache_manifest: Temprorary cache file with sampled data.
            batch_size: Batch size used for during inference.
            num_workers: (int) number of workers for DataLoader
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

        dl_config = {
            'manifest_filepath': cache_manifest,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': True,
            'channel_selector': None,
            'use_start_end_token': False,
        }
        
        dataloader = self._setup_pseudo_label_dataloader_from_config(dl_config)

        self.preprocessor.featurizer.dither = 0.0
        self.preprocessor.featurizer.pad_to = 0
        sample_idx = 0
        
        for test_batch in tqdm(dataloader, desc="Transcribing"):
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
                        target = target_transcripts[sample_idx + beams_idx ]
                        if target and restore_pc:
                            target_split_w = target.split()
                            wer_dist_min = 1000
                            min_pred_text = ""
                            for _, candidate in enumerate(beams): 
                                print("will do pc restore")
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
    
    def _setup_pseudo_label_dataloader_from_config(self, config: Dict):
    
        dataset = audio_to_text_dataset.get_bpe_dataset(config=config, tokenizer=self.tokenizer, augmentor=None)
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
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )
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
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'channel_selector': config.get('channel_selector', None),
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
        ctc_decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Directory path to tokenizer or a config for a new tokenizer (if the tokenizer type is `agg`)
            new_tokenizer_type: Type of tokenizer. Can be either `agg`, `bpe` or `wpe`.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            ctc_decoding_cfg: A config for auxiliary CTC decoding, which is optional and can be used to change the decoding type.

        Returns: None

        """
        if isinstance(new_tokenizer_dir, DictConfig):
            if new_tokenizer_type == 'agg':
                new_tokenizer_cfg = new_tokenizer_dir
            else:
                raise ValueError(
                    f'New tokenizer dir should be a string unless the tokenizer is `agg`, but this tokenizer type is: {new_tokenizer_type}'
                )
        else:
            new_tokenizer_cfg = None

        if new_tokenizer_cfg is not None:
            tokenizer_cfg = new_tokenizer_cfg
        else:
            if not os.path.isdir(new_tokenizer_dir):
                raise NotADirectoryError(
                    f'New tokenizer dir must be non-empty path to a directory. But I got: {new_tokenizer_dir}'
                )

            if new_tokenizer_type.lower() not in ('bpe', 'wpe'):
                raise ValueError(f'New tokenizer type must be either `bpe` or `wpe`')

            tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        # Setup the tokenizer
        self._setup_tokenizer(tokenizer_cfg)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        joint_config = self.joint.to_config_dict()
        new_joint_config = copy.deepcopy(joint_config)
        if self.tokenizer_type == "agg":
            new_joint_config["vocabulary"] = ListConfig(vocabulary)
        else:
            new_joint_config["vocabulary"] = ListConfig(list(vocabulary.keys()))

        new_joint_config['num_classes'] = len(vocabulary)
        del self.joint
        self.joint = EncDecHybridRNNTCTCBPEModel.from_config_dict(new_joint_config)

        decoder_config = self.decoder.to_config_dict()
        new_decoder_config = copy.deepcopy(decoder_config)
        new_decoder_config.vocab_size = len(vocabulary)
        del self.decoder
        self.decoder = EncDecHybridRNNTCTCBPEModel.from_config_dict(new_decoder_config)

        del self.loss
        self.loss = RNNTLoss(num_classes=self.joint.num_classes_with_blank - 1)

        if decoding_cfg is None:
            # Assume same decoding config as before
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = RNNTBPEDecoding(
            decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Update config
        with open_dict(self.cfg.joint):
            self.cfg.joint = new_joint_config

        with open_dict(self.cfg.decoder):
            self.cfg.decoder = new_decoder_config

        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed tokenizer of the RNNT decoder to {self.joint.vocabulary} vocabulary.")

        # set up the new tokenizer for the CTC decoder
        if hasattr(self, 'ctc_decoder'):
            ctc_decoder_config = copy.deepcopy(self.ctc_decoder.to_config_dict())
            # sidestepping the potential overlapping tokens issue in aggregate tokenizers
            if self.tokenizer_type == "agg":
                ctc_decoder_config.vocabulary = ListConfig(vocabulary)
            else:
                ctc_decoder_config.vocabulary = ListConfig(list(vocabulary.keys()))

            decoder_num_classes = ctc_decoder_config['num_classes']
            # Override number of classes if placeholder provided
            logging.info(
                "\nReplacing old number of classes ({}) with new number of classes - {}".format(
                    decoder_num_classes, len(vocabulary)
                )
            )
            ctc_decoder_config['num_classes'] = len(vocabulary)

            del self.ctc_decoder
            self.ctc_decoder = EncDecHybridRNNTCTCBPEModel.from_config_dict(ctc_decoder_config)
            del self.ctc_loss
            self.ctc_loss = CTCLoss(
                num_classes=self.ctc_decoder.num_classes_with_blank - 1,
                zero_infinity=True,
                reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
            )

            if ctc_decoding_cfg is None:
                # Assume same decoding config as before
                ctc_decoding_cfg = self.cfg.aux_ctc.decoding

            # Assert the decoding config with all hyper parameters
            ctc_decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
            ctc_decoding_cls = OmegaConf.create(OmegaConf.to_container(ctc_decoding_cls))
            ctc_decoding_cfg = OmegaConf.merge(ctc_decoding_cls, ctc_decoding_cfg)

            self.ctc_decoding = CTCBPEDecoding(decoding_cfg=ctc_decoding_cfg, tokenizer=self.tokenizer)

            self.ctc_wer = WER(
                decoding=self.ctc_decoding,
                use_cer=self.cfg.aux_ctc.get('use_cer', False),
                log_prediction=self.cfg.get("log_prediction", False),
                dist_sync_on_step=True,
            )

            # Update config
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoder = ctc_decoder_config

            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg

            logging.info(f"Changed tokenizer of the CTC decoder to {self.ctc_decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig = None, decoder_type: str = None):
        """
        Changes decoding strategy used during RNNT decoding process.
        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            decoder_type: (str) Can be set to 'rnnt' or 'ctc' to switch between appropriate decoder in a
                model having both RNN-T and CTC decoders. Defaults to None, in which case RNN-T decoder is
                used. If set to 'ctc', it raises error if 'ctc_decoder' is not an attribute of the model.
        """
        if decoder_type is None or decoder_type == 'rnnt':
            if decoding_cfg is None:
                # Assume same decoding config as before
                logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
                decoding_cfg = self.cfg.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

            self.decoding = RNNTBPEDecoding(
                decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
            )

            self.wer = WER(
                decoding=self.decoding,
                batch_dim_index=self.wer.batch_dim_index,
                use_cer=self.wer.use_cer,
                log_prediction=self.wer.log_prediction,
                dist_sync_on_step=True,
            )

            # Setup fused Joint step
            if self.joint.fuse_loss_wer or (
                self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
            ):
                self.joint.set_loss(self.loss)
                self.joint.set_wer(self.wer)

            self.joint.temperature = decoding_cfg.get('temperature', 1.0)

            # Update config
            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            self.cur_decoder = "rnnt"
            logging.info(f"Changed decoding strategy of the RNNT decoder to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

        elif decoder_type == 'ctc':
            if not hasattr(self, 'ctc_decoding'):
                raise ValueError("The model does not have the ctc_decoding module and does not support ctc decoding.")
            if decoding_cfg is None:
                # Assume same decoding config as before
                logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
                decoding_cfg = self.cfg.aux_ctc.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

            self.ctc_decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer)

            self.ctc_wer = WER(
                decoding=self.ctc_decoding,
                use_cer=self.ctc_wer.use_cer,
                log_prediction=self.ctc_wer.log_prediction,
                dist_sync_on_step=True,
            )

            self.ctc_decoder.temperature = decoding_cfg.get('temperature', 1.0)

            # Update config
            with open_dict(self.cfg.aux_ctc.decoding):
                self.cfg.aux_ctc.decoding = decoding_cfg

            self.cur_decoder = "ctc"
            logging.info(
                f"Changed decoding strategy of the CTC decoder to \n{OmegaConf.to_yaml(self.cfg.aux_ctc.decoding)}"
            )
        else:
            raise ValueError(f"decoder_type={decoder_type} is not supported. Supported values: [ctc,rnnt]")

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_en_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_de_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_it_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_it_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_fastconformer_hybrid_large_pc/versions/1.20.0/files/stt_it_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_es_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_hr_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_hr_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_hr_fastconformer_hybrid_large_pc/versions/1.21.0/files/FastConformer-Hybrid-Transducer-CTC-BPE-v256-averaged.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ua_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ua_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ua_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_ua_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_pl_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_pl_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_pl_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_pl_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_by_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_by_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_by_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_by_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ru_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ru_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_ru_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_fr_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_fr_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_multilingual_fastconformer_hybrid_large_pc_blend_eu",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_multilingual_fastconformer_hybrid_large_pc_blend_eu",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_multilingual_fastconformer_hybrid_large_pc_blend_eu/versions/1.21.0/files/stt_multilingual_fastconformer_hybrid_large_pc_blend_eu.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_multilingual_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_multilingual_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_multilingual_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_multilingual_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_80ms",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_80ms",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_80ms/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_80ms.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_480ms",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_480ms",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_480ms/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_480ms.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_1040ms",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_1040ms",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_1040ms/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_1040ms.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_multi",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_multi",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_multi/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
        )
        results.append(model)

        return results
