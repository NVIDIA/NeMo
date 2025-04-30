# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Dict, List, Optional, Union, Tuple
import math
import numpy as np
import tempfile
import json
from tqdm import tqdm

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.data.audio_to_text_lhotse_target_language import LhotseSpeechToTextBpeDatasetTgtLangID
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.metrics.bleu import BLEU
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, InterCTCMixin, TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import TranscriptionReturnType
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType

from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTBPEDecodingConfig
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging, model_utils
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, SpectrogramType, LabelsType, AcousticEncodedRepresentation


class EncDecHybridRNNTCTCBPEModelTgtLangID(EncDecHybridRNNTCTCModel, ASRBPEMixin):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss and subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        self._GLOBAL_LANG_MAP= {
            # Group 1: 
            'en-US': 0,   'en-GB': 1,   
            'es-ES': 2,   'es-US': 3,   # Spanish variants
            'zh-CN': 4,   'zh-TW': 5,   # Chinese variants
            'hi-IN': 6,   'ar-AR': 7,   # Hindi & Arabic
            'fr-FR': 8,   'de-DE': 9,   # French & German
            'ja-JP': 10,  'ru-RU': 11,  # Japanese & Russian
            'pt-BR': 12,  'pt-PT': 13,  # Portuguese variants
            'ko-KR': 14,  'it-IT': 15,  # Korean & Italian

            # Group 2: 
            'nl-NL': 16,  'pl-PL': 17,  
            'tr-TR': 18,  'uk-UA': 19,
            'ro-RO': 20,  'el-GR': 21,
            'cs-CZ': 22,  'hu-HU': 23,
            'sv-SE': 24,  'da-DK': 25,
            'fi-FI': 26,  'no-NO': 27,
            'sk-SK': 28,  'hr-HR': 29,
            'bg-BG': 30,  'lt-LT': 31,

            # Group 3: 
            'th-TH': 32,  'vi-VN': 33,
            'id-ID': 34,  'ms-MY': 35,
            'bn-IN': 36,  'ur-PK': 37,
            'fa-IR': 38,  'ta-IN': 39,
            'te-IN': 40,  'mr-IN': 41,
            'gu-IN': 42,  'kn-IN': 43,
            'ml-IN': 44,  'si-LK': 45,
            'ne-NP': 46,  'km-KH': 47,

            # Group 4: 
            'sw-KE': 48,  'am-ET': 49,
            'ha-NG': 50,  'zu-ZA': 51,
            'yo-NG': 52,  'ig-NG': 53,
            'af-ZA': 54,  'rw-RW': 55,
            'so-SO': 56,  'ny-MW': 57,
            'ln-CD': 58,  'or-KE': 59,


            # Group 5: 
            'he-IL': 64,  'ku-TR': 65,
            'az-AZ': 66,  'ka-GE': 67,
            'hy-AM': 68,  'uz-UZ': 69,
            'tg-TJ': 70,  'ky-KG': 71,

            'qu-PE': 80,  'ay-BO': 81,
            'gn-PY': 82,  'nah-MX': 83,


            # Group 7: 
            'mi-NZ': 96,  'haw-US': 97,
            'sm-WS': 98,  'to-TO': 99}

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

        super().__init__(cfg=cfg, trainer=trainer)

        if cfg.get("initialize_target_lang_id_concatination", False):
            self.initialize_target_lang_id_concatination()

    def initialize_target_lang_id_concatination(self):
        """Initialize model components for target language ID concatenation."""
        print("target language model has been initalized")

        # Setup decoding object
        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
        )

        # Setup wer object
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self.cfg.get('use_cer', False),
            log_prediction=self.cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # Setup bleu object
        self.bleu = BLEU(decoding=self.decoding, tokenize=self.cfg.get('bleu_tokenizer', "13a"), log_prediction=True)

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

        if hasattr(self.cfg, 'initialize_target_lang_id_concatination'):
            self.concat = True
            self.num_langs = self._cfg.model_defaults.get('num_langs', 128)

            # Setup normalization
            self.norm = self._cfg.get('norm', None)
            if self._cfg.get('norm') == 'ln':
                self.asr_norm = torch.nn.LayerNorm(self._cfg.model_defaults.enc_hidden)
                self.lang_norm = torch.nn.LayerNorm(self.num_langs)

            # Setup projection layers
            proj_in_size = self.num_langs + self._cfg.model_defaults.enc_hidden
            proj_out_size = self._cfg.model_defaults.enc_hidden

            self.lang_kernal = torch.nn.Sequential(
                torch.nn.Linear(proj_in_size, proj_out_size * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_out_size * 2, proj_out_size),
            )

    def _setup_dataloader_from_config(self, config: Optional[Dict]):

        if config.get("use_lhotse"):
            if 'initialize_target_lang_id_concatination' in self.cfg:
                dataset = LhotseSpeechToTextBpeDatasetTgtLangID(tokenizer=self.tokenizer, cfg=config)
            else:
                dataset = LhotseSpeechToTextBpeDataset(tokenizer=self.tokenizer)
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=dataset,
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

    # this function is taken from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length        
    def get_hidden_length_from_sample_length(self,
        num_samples: int, 
        num_sample_per_mel_frame: int = 160, 
        num_mel_frame_per_asr_frame: int = 8
    ) -> int:
        """ 
        Calculate the hidden length from the given number of samples.

        This function computes the number of frames required for a given number of audio samples,
        considering the number of samples per mel frame and the number of mel frames per ASR frame.

        Parameters:
            num_samples (int): The total number of audio samples.
            num_sample_per_mel_frame (int, optional): The number of samples per mel frame. Default is 160.
            num_mel_frame_per_asr_frame (int, optional): The number of mel frames per ASR frame. Default is 8.

        Returns:
            hidden_length (int): The calculated hidden length in terms of the number of frames.
        """
        mel_frame_count = math.ceil((num_samples + 1) / num_sample_per_mel_frame)
        hidden_length = math.ceil(mel_frame_count / num_mel_frame_per_asr_frame)
        return int(hidden_length)
        
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

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            print("'manifest_filepath' in config")
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            # import pdb
            # pdb.set_trace()
            print(manifest_filepath)
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.joint.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_lhotse': True,
            'use_bucketing': False,
            'drop_last': False,
            'initialize_target_lang_id_concatination': True,
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")


        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    @torch.no_grad()
    def transcribe(
        self,
        manifest_filepath: str,
        paths2audio_files: List[str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        # channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
    ) -> Tuple[List[str], Optional[List['Hypothesis']]]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:

            paths2audio_files: (a list) of paths to audio files. \
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
        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        # We will store transcriptions here
        hypotheses = []
        all_hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0

            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.decoder.freeze()
            self.joint.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'manifest_filepath': manifest_filepath,
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                    # 'channel_selector': channel_selector,
                }

                if augmentor:
                    config['augmentor'] = augmentor

                temporary_datalayer = self._setup_transcribe_dataloader(config)

                for test_batch in tqdm(temporary_datalayer, desc="Transcribing", disable=(not verbose)):
                    encoded, encoded_len = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device), target_lang_id=test_batch[4].to(device)
                    )
                    best_hyp, all_hyp = self.decoding.rnnt_decoder_predictions_tensor(
                        encoded,
                        encoded_len,
                        return_hypotheses=return_hypotheses,
                        partial_hypotheses=partial_hypothesis,
                    )

                    hypotheses += best_hyp
                    if all_hyp is not None:
                        all_hypotheses += all_hyp
                    else:
                        all_hypotheses += best_hyp

                    del encoded
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value

            logging.set_verbosity(logging_level)
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
                self.joint.unfreeze()
        return hypotheses, all_hypotheses
    # def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
    #     """
    #     Setup function for a temporary data loader which wraps the provided audio file.

    #     Args:
    #         config: A python dictionary which contains the following keys:
    #         paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
    #             Recommended length per file is between 5 and 25 seconds.
    #         batch_size: (int) batch size to use during inference. \
    #             Bigger will result in better throughput performance but would use more memory.
    #         temp_dir: (str) A temporary directory where the audio manifest is temporarily
    #             stored.

    #     Returns:
    #         A pytorch DataLoader for the given audio file(s).
    #     """
    #     if 'manifest_filepath' in config:
    #         print("'manifest_filepath' in config")
    #         manifest_filepath = config['manifest_filepath']
    #         batch_size = config['batch_size']
    #     else:
    #         manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
    #         print(manifest_filepath)
    #         batch_size = min(config['batch_size'], len(config['paths2audio_files']))

    #     dl_config = {
    #         'manifest_filepath': manifest_filepath,
    #         'sample_rate': self.preprocessor._sample_rate,
    #         'labels': self.joint.vocabulary,
    #         'batch_size': batch_size,
    #         'trim_silence': False,
    #         'shuffle': False,
    #         'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
    #         'pin_memory': True,
    #         'use_lhotse': True,
    #         'use_bucketing': False,
    #         'drop_last': False,
    #         'initialize_target_lang_id_concatination': True,
    #     }

    #     if config.get("augmentor"):
    #         dl_config['augmentor'] = config.get("augmentor")

    #     temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
    #     return temporary_datalayer

    # @torch.no_grad()
    # def transcribe(
    #     self,
    #     audio: List[str],
    #     batch_size: int = 4,
    #     return_hypotheses: bool = False,
    #     partial_hypothesis: Optional[List['Hypothesis']] = None,
    #     num_workers: int = 0,
    #     # channel_selector: Optional[ChannelSelectorType] = None,
    #     augmentor: DictConfig = None,
    #     verbose: bool = True,
    #     target_lang: str = None,
    #     # override_config: Optional[TranscribeConfig] = None,
    # ) -> TranscriptionReturnType:
    #     """
    #     Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

    #     Args:
    #         audio: (a single or list) of paths to audio files or a np.ndarray audio array.
    #             Can also be a dataloader object that provides values that can be consumed by the model.
    #             Recommended length per file is between 5 and 25 seconds. \
    #             But it is possible to pass a few hours long file if enough GPU memory is available.
    #         batch_size: (int) batch size to use during inference. \
    #             Bigger will result in better throughput performance but would use more memory.
    #         return_hypotheses: (bool) Either return hypotheses or text
    #             With hypotheses can do some postprocessing like getting timestamp or rescoring
    #         num_workers: (int) number of workers for DataLoader
    #         channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels 
    #             from multi-channel audio. If set to `'average'`, it performs averaging across channels. 
    #             Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
    #         augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
    #         verbose: (bool) whether to display tqdm progress bar
    #         target_lang: (str) Target language code (e.g., "en-US", "de-DE") for transcription/translation
    #         override_config: (Optional[TranscribeConfig]) Configuration overrides for transcription

    #     Returns:
    #         Returns a tuple of 2 items -
    #         * A list of greedy transcript texts / Hypothesis
    #         * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
    #     """
    #     if self.cur_decoder not in ["ctc", "rnnt"]:
    #         raise ValueError(
    #             f"{self.cur_decoder} is not supported for cur_decoder. Supported values are ['ctc', 'rnnt']"
    #         )

    #     # Create or update the override config with target language info
    #     config_dict = {}
    #     if override_config is not None:
    #         config_dict = {k: getattr(override_config, k) for k in override_config.__dict__ 
    #                     if not k.startswith('_')}
        
    #     # Add target language to config if provided
    #     if target_lang is not None:
    #         config_dict['target_lang'] = target_lang

    #     # Create a new TranscribeConfig or update the existing one
    #     if config_dict and override_config is None:
    #         override_config = TranscribeConfig(**config_dict)
    #     elif config_dict:
    #         for k, v in config_dict.items():
    #             setattr(override_config, k, v)

    #     return super().transcribe(
    #         audio=audio,
    #         batch_size=batch_size,
    #         return_hypotheses=return_hypotheses,
    #         partial_hypothesis=partial_hypothesis,
    #         num_workers=num_workers,
    #         channel_selector=channel_selector,
    #         augmentor=augmentor,
    #         verbose=verbose,
    #         override_config=override_config,
    #     )

    # def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
    #     """
    #     Forward pass for transcription that handles target language codes
    #     """
    #     # Create language ID tensor if target language is specified
    #     lang_id_tensor = None
        
    #     # Check if we have a target language specified
    #     if hasattr(trcfg, 'target_lang') and trcfg.target_lang is not None:
    #         # Convert string language code to numeric ID
    #         try:
    #             lang_id = self._GLOBAL_LANG_MAP[trcfg.target_lang]
                
    #             # Create a tensor for the language ID
    #             batch_size = batch[0].size(0)
    #             device = batch[0].device
                
    #             # Get the language embedding dimension from the model
    #             embedding_dim = self.num_langs if hasattr(self, 'num_langs') else 128
                
    #             # Calculate encoder output length based on input audio
    #             num_samples = batch[0].size(-1)
    #             hidden_length = self.get_hidden_length_from_sample_length(
    #                 num_samples,
    #                 num_sample_per_mel_frame=160,  # Default value, adjust as needed
    #                 num_mel_frame_per_asr_frame=8  # Default value, adjust as needed
    #             )
                
    #             # Create the language ID tensor (batch_size, seq_len, embedding_dim)
    #             lang_id_tensor = torch.zeros(batch_size, hidden_length, embedding_dim, device=device)
                
    #             # Set the corresponding language ID to 1 for all time steps
    #             lang_id_tensor[:, :, lang_id] = 1
                
    #             print(f"Using target language: {trcfg.target_lang} (ID: {lang_id})")
    #         except KeyError:
    #             print(f"Warning: Unknown language code: {trcfg.target_lang}. Proceeding without language ID.")
        
    #     if self.cur_decoder == "rnnt":
    #         # Forward pass with the language ID tensor
    #         if lang_id_tensor is not None:
    #             encoded, encoded_len = self.forward(
    #                 input_signal=batch[0], 
    #                 input_signal_length=batch[1],
    #                 target_lang_id=lang_id_tensor
    #             )
                
    #             # Apply decoder
    #             best_hyp, all_hyp = self.decoding.rnnt_decoder_predictions_tensor(
    #                 encoder_output=encoded,
    #                 encoded_lengths=encoded_len,
    #                 return_hypotheses=hasattr(trcfg, 'return_hypotheses') and trcfg.return_hypotheses,
    #                 partial_hypotheses=hasattr(trcfg, 'partial_hypothesis') and trcfg.partial_hypothesis,
    #             )
                
    #             return {"best_hyp": best_hyp, "all_hyp": all_hyp}
    #         else:
    #             # No language ID specified, call the parent method
    #             return super()._transcribe_forward(batch, trcfg)
    #     else:
    #         # CTC Path
    #         if lang_id_tensor is not None:
    #             encoded, encoded_len = self.forward(
    #                 input_signal=batch[0], 
    #                 input_signal_length=batch[1],
    #                 target_lang_id=lang_id_tensor
    #             )
    #         else:
    #             encoded, encoded_len = self.forward(
    #                 input_signal=batch[0], 
    #                 input_signal_length=batch[1]
    #             )
            
    #         logits = self.ctc_decoder(encoder_output=encoded)
    #         output = dict(logits=logits, encoded_len=encoded_len)
            
    #         del encoded
    #         return output

    # def _transcribe_output_processing(
    #     self, outputs, trcfg: TranscribeConfig
    # ) -> Tuple[List['Hypothesis'], List['Hypothesis']]:
    #     if self.cur_decoder == "rnnt":
    #         return super()._transcribe_output_processing(outputs, trcfg)

    #     # CTC Path
    #     logits = outputs.pop('logits')
    #     encoded_len = outputs.pop('encoded_len')

    #     best_hyp, all_hyp = self.ctc_decoding.ctc_decoder_predictions_tensor(
    #         logits,
    #         encoded_len,
    #         return_hypotheses=trcfg.return_hypotheses,
    #     )
    #     logits = logits.cpu()

    #     if trcfg.return_hypotheses:
    #         # dump log probs per file
    #         for idx in range(logits.shape[0]):
    #             best_hyp[idx].y_sequence = logits[idx][: encoded_len[idx]]
    #             if best_hyp[idx].alignments is None:
    #                 best_hyp[idx].alignments = best_hyp[idx].y_sequence

    #     # DEPRECATED?
    #     # if logprobs:
    #     #     for logit, elen in zip(logits, encoded_len):
    #     #         logits_list.append(logit[:elen])

    #     del logits, encoded_len

    #     hypotheses = []
    #     all_hypotheses = []

    #     hypotheses += best_hyp
    #     if all_hyp is not None:
    #         all_hypotheses += all_hyp
    #     else:
    #         all_hypotheses += best_hyp

    #     return (hypotheses, all_hypotheses)

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
                raise ValueError('New tokenizer type must be either `bpe` or `wpe`')

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
        self.joint = EncDecHybridRNNTCTCBPEModelTgtLangID.from_config_dict(new_joint_config)

        decoder_config = self.decoder.to_config_dict()
        new_decoder_config = copy.deepcopy(decoder_config)
        new_decoder_config.vocab_size = len(vocabulary)
        del self.decoder
        self.decoder = EncDecHybridRNNTCTCBPEModelTgtLangID.from_config_dict(new_decoder_config)

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
            decoding_cfg=decoding_cfg,
            decoder=self.decoder,
            joint=self.joint,
            tokenizer=self.tokenizer,
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
            self.ctc_decoder = EncDecHybridRNNTCTCBPEModelTgtLangID.from_config_dict(ctc_decoder_config)
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
                decoding_cfg=decoding_cfg,
                decoder=self.decoder,
                joint=self.joint,
                tokenizer=self.tokenizer,
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
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
            'target_lang_id': NeuralType(('B', 'T', 'D'), LabelsType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        target_lang_id=None,
    ):
        """
        Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `training_step` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

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
            A tuple of 2 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = torch.transpose(encoded, 1, 2)  # B * D * T -> B * T * D

        if self.concat:
            if target_lang_id.shape[1] > encoded.shape[1]:
                target_lang_id = target_lang_id[:, : encoded.shape[1], :]

            # Concatenate encoded states with language ID
            concat_enc_states = torch.cat([encoded, target_lang_id], dim=-1)

            # Apply joint projection
            encoded = self.lang_kernal(concat_enc_states)

        encoded = torch.transpose(encoded, 1, 2)  # B * T * D -> B * D * T
        return encoded, encoded_len

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len, target_lang_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len, target_lang_id=target_lang_id
            )
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

            # Reset access registry
            # from RNNT training
            # if AccessMixin.is_access_enabled():
            #     AccessMixin.reset_registry(self)

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

            # Reset access registry
            # from RNNT training
            # if AccessMixin.is_access_enabled():
            #     AccessMixin.reset_registry(self)

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
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO: add support for CTC decoding
        signal, signal_len, transcript, transcript_len, sample_id, target_lang_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len, target_lang_id=target_lang_id
            )
        del signal

        best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_pass(self, batch, batch_idx, dataloader_idx):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len, target_lang_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len, target_lang_id=target_lang_id
            )
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

        # CTC WER calculation
        self.ctc_wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        # BLEU score calculation
        self.bleu.update(
            predictions=encoded, predictions_lengths=encoded_len, targets=transcript, targets_lengths=transcript_len
        )
        bleu_metrics = self.bleu.compute(return_all_metrics=True, prefix="val_")
        tensorboard_logs.update(
            {
                'val_bleu_num': bleu_metrics['val_bleu_num'],
                'val_bleu_denom': bleu_metrics['val_bleu_denom'],
                'val_bleu_pred_len': bleu_metrics['val_bleu_pred_len'],
                'val_bleu_target_len': bleu_metrics['val_bleu_target_len'],
                'val_bleu': bleu_metrics['val_bleu'],
            }
        )
        self.bleu.reset()

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Inter-CTC losses and additional logging
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
        if AccessMixin.is_access_enabled():
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
        # Calculate validation loss if required
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}

        # Calculate WER
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}

        # Calculate CTC WER if applicable
        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['val_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['val_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['val_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        # Calculate BLEU score
        bleu_num = torch.stack([x['val_bleu_num'] for x in outputs]).sum(dim=0)
        bleu_denom = torch.stack([x['val_bleu_denom'] for x in outputs]).sum(dim=0)
        bleu_pred_len = torch.stack([x['val_bleu_pred_len'] for x in outputs]).sum(dim=0)
        bleu_target_len = torch.stack([x['val_bleu_target_len'] for x in outputs]).sum(dim=0)

        val_bleu = self.bleu._compute_bleu(bleu_pred_len, bleu_target_len, bleu_num, bleu_denom)
        tensorboard_logs['val_bleu'] = val_bleu

        # Finalize and log metrics
        metrics = {**val_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")

        return metrics


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

    @property
    def bleu(self):
        return self._bleu

    @bleu.setter
    def bleu(self, bleu):
        self._bleu = bleu
