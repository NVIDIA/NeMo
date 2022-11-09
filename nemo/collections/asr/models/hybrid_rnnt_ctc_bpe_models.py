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
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.models.rnnt_models import EncDecRNNTBPEModel
from nemo.collections.asr.metrics.wer_bpe import WERBPE, CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.utils import logging, model_utils


class EncDecHybridRNNTCTCModel(EncDecRNNTBPEModel):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss and subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        super().__init__(cfg=cfg, trainer=trainer)
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # setup auxiliary CTC decoder if needed
        if 'aux_ctc' in cfg:
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

            # Setup CTC decoding
            ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
            if ctc_decoding_cfg is None:
                ctc_decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
                with open_dict(self.cfg.ctc_decoding):
                    self.cfg.aux_ctc.decoding = ctc_decoding_cfg
            self.ctc_decoding = CTCBPEDecoding(self.cfg.ctc_decoding.decoder, tokenizer=self.tokenizer)

            # Setup CTC WER
            self.ctc_wer = WERBPE(
                decoding=self.ctc_decoding,
                use_cer=self.cfg.ctc_decoding.get('use_cer', False),
                dist_sync_on_step=True,
                log_prediction=self.cfg.get("log_prediction", False),
            )

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
        super().__init__(new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type=new_tokenizer_type, decoding_cfg=decoding_cfg)

        # set up CTC decoder if required
        if ctc_decoding_cfg is not None: # or self.ctc_loss_weight > 0:
            if hasattr(self, 'ctc_decoding'):
                decoder_config = self.ctc_decoding.to_config_dict()
                new_decoder_config = copy.deepcopy(decoder_config)

                del self.ctc_decoding
                del self.ctc_loss
            else:
                new_decoder_config = self.cfg.ctc_decoding.decoder

            # sidestepping the potential overlapping tokens issue in aggregate tokenizers
            if self.tokenizer_type == "agg":
                new_decoder_config.vocabulary = ListConfig(vocabulary)
            else:
                new_decoder_config.vocabulary = ListConfig(list(vocabulary.keys()))
            new_decoder_config['num_classes'] = len(vocabulary)

            self.ctc_decoder = EncDecCTCModelBPE.from_config_dict(new_decoder_config)
            self.ctc_loss = CTCLoss(
                num_classes=self.ctc_decoding.num_classes_with_blank - 1,
                zero_infinity=True,
                reduction=self.cfg.ctc_decoding.get("ctc_reduction", "mean_batch"),
            )

            if ctc_decoding_cfg is None:
                ctc_decoding_cfg = self.cfg.ctc_decoding.get('decoding', None)
            if ctc_decoding_cfg is None:
                ctc_decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
                with open_dict(self.cfg.ctc_decoding):
                    self.cfg.ctc_decoding.decoder = ctc_decoding_cfg

            # Assert the decoding config with all hyper parameters
            ctc_decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
            ctc_decoding_cls = OmegaConf.create(OmegaConf.to_container(ctc_decoding_cls))
            ctc_decoding_cfg = OmegaConf.merge(ctc_decoding_cls, ctc_decoding_cfg)

            self.ctc_decoding = CTCBPEDecoding(decoding_cfg=ctc_decoding_cfg, tokenizer=self.tokenizer)

            self.ctc_wer = WERBPE(
                decoding=self.ctc_decoding,
                use_cer=self.cfg.ctc_decoding.get('use_cer', False),
                log_prediction=self.cfg.get("log_prediction", False),
                dist_sync_on_step=True,
            )

            # Update config
            with open_dict(self.cfg.ctc_decoding.decoder):
                self.cfg.ctc_decoding.decoder = new_decoder_config

            with open_dict(self.cfg.ctc_decoding.decoding):
                self.cfg.ctc_decoding.decoding = ctc_decoding_cfg

    def change_decoding_strategy(self, decoding_cfg: DictConfig, decoder_type: str = None):
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
            super().__init__(decoding_cfg=decoding_cfg)
        else:
            assert decoder_type == 'ctc' and hasattr(self, 'ctc_decoding')
            if decoding_cfg is None:
                # Assume same decoding config as before
                logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
                decoding_cfg = self.cfg.ctc_decoding.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

            self.ctc_decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer)

            self.ctc_wer = WERBPE(
                decoding=self.ctc_decoding,
                use_cer=self.ctc_wer.use_cer,
                log_prediction=self.ctc_wer.log_prediction,
                dist_sync_on_step=True,
            )

            # Update config
            with open_dict(self.cfg.ctc):
                self.cfg.ctc_decoding.decoding = decoding_cfg

            self.use_rnnt_decoder = False
            logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.ctc_decoding.decoding)}")
