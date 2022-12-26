import copy
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.data.text_to_text import TextOrAudioToTextBatch, TextToTextBatch, TextToTextDataset
from nemo.collections.asr.models import ASRModel, EncDecCTCModelBPE, EncDecRNNTBPEModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.preprocessing.features import clean_spectrogram_batch, normalize_batch
from nemo.collections.asr.parts.submodules.batchnorm import replace_bn_with_fused_bn_all
from nemo.collections.tts.models import FastPitchModel
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum


def _fuse_bn_in_conformer(asr_model: ASRModel):
    logging.info("Replacing BatchNorm with Fused BatchNorm")
    if not hasattr(asr_model, "encoder"):
        raise NotImplementedError("No encoder found in ASR Model, replacement not supported")
    if not isinstance(asr_model.encoder, ConformerEncoder):
        raise NotImplementedError(f"Unsupported encoder type: {type(asr_model.encoder)}")
    replace_bn_with_fused_bn_all(asr_model.encoder)
    if "conv_norm_type" not in asr_model.cfg.encoder:
        # old CTC models from NGC don't have such param
        logging.warning("conv_norm_type not in encoder config, adding parameter")
        OmegaConf.set_struct(asr_model.cfg, False)
        asr_model.cfg.encoder.conv_norm_type = "fused_batch_norm"
        OmegaConf.set_struct(asr_model.cfg, True)
    else:
        asr_model.cfg.encoder.conv_norm_type = "fused_batch_norm"


class ASRWithTTSModel(ASRModel):
    asr_model: Union[EncDecRNNTBPEModel, EncDecCTCModelBPE]
    tts_model: FastPitchModel

    class ASRModelTypes(PrettyStrEnum):
        RNNT_BPE = "rnnt_bpe"
        CTC_BPE = "ctc_bpe"

        @classmethod
        def from_asr_model(cls, model: Any):
            if isinstance(model, EncDecRNNTBPEModel):
                return cls.RNNT_BPE
            if isinstance(model, EncDecCTCModelBPE):
                return cls.CTC_BPE
            raise ValueError(f"Unsupported model type: {type(model)}")

        def get_asr_cls(self):
            if self == self.RNNT_BPE:
                return EncDecRNNTBPEModel
            if self == self.CTC_BPE:
                return EncDecCTCModelBPE
            raise NotImplementedError(f"Not implemented for value {self.value}")

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        config should contain
        :param cfg:
        :param trainer:
        """
        self._full_init_guard = False
        cfg = copy.deepcopy(cfg)
        OmegaConf.set_struct(cfg, False)
        # avoid dataset and optim setup here
        train_ds_cfg = cfg.pop("train_ds", None)
        validation_ds_cfg = cfg.pop("validation_ds", None)
        test_ds_cfg = cfg.pop("test_ds", None)
        optim_cfg = cfg.pop("optim", None)

        super().__init__(cfg, trainer=trainer)

        # tts model
        if cfg.tts_model_path is not None:
            self.tts_model = FastPitchModel.restore_from(cfg.tts_model_path, map_location=torch.device("cpu"))
            self.cfg.tts_model_path = None
            self.cfg.tts_model = self.tts_model.cfg
        else:
            assert cfg.tts_model is not None
            self.tts_model = ModelPT.from_config_dict(cfg.tts_model)
        self.register_submodule_artifacts(self.tts_model, "tts_model")
        self.tts_model.freeze()  # tts model should be always frozen

        if cfg.asr_model_path is not None:
            self.asr_model = ASRModel.restore_from(cfg.asr_model_path, map_location=torch.device("cpu"))
            self.asr_model_type = self.ASRModelTypes.from_asr_model(self.asr_model)
            self.cfg.asr_model_type = str(self.asr_model_type)
        else:
            self.asr_model_type = self.ASRModelTypes(cfg.asr_model_type)  # convert to enum
            self.asr_model = self.asr_model_type.get_asr_cls()(cfg.asr_model)  # instantiate
            self.cfg.asr_model = self.asr_model.cfg
        self.register_submodule_artifacts(self.asr_model, "asr_model")

        # replace BatchNorm with FusedBatchNorm
        if cfg.get("asr_model_fuse_bn"):
            _fuse_bn_in_conformer(self.asr_model)
            self.cfg.asr_model = self.asr_model.cfg
            cfg.asr_model_fuse_bn = False  # no need to fuse anymore

        if cfg.enhancer_model_path is not None:
            # ToDo: add enhancer support after https://github.com/NVIDIA/NeMo/pull/5565
            raise NotImplementedError

        self._full_init_guard = True

        if optim_cfg:
            OmegaConf.set_struct(self.cfg, False)
            self.cfg.optim = optim_cfg
            self.setup_optimization(optim_config=optim_cfg)
        if train_ds_cfg:
            OmegaConf.set_struct(self.cfg, False)
            self.cfg.train_ds = train_ds_cfg
            self.setup_training_data(train_data_config=train_ds_cfg)
        if validation_ds_cfg:
            OmegaConf.set_struct(self.cfg, False)
            self.cfg.validation_ds = validation_ds_cfg
            self.setup_multiple_validation_data(val_data_config=validation_ds_cfg)
        if test_ds_cfg:
            OmegaConf.set_struct(self.cfg, False)
            self.cfg.test_ds = test_ds_cfg
            self.setup_test_data(test_data_config=test_ds_cfg)
        OmegaConf.set_struct(self.cfg, True)

    @classmethod
    def from_asr_config(
        cls,
        asr_cfg: DictConfig,
        asr_model_type: Union[str, ASRModelTypes],
        tts_model_path: Union[str, Path],
        trainer: Trainer = None,
    ):
        """
        Method to construct model from ASR config for training from scratch
        :param asr_cfg:
        :param asr_model_type:
        :param tts_model_path:
        :param trainer:
        :return:
        """
        model_type = cls.ASRModelTypes(asr_model_type)
        cfg = DictConfig(
            dict(
                asr_model_path=None,
                asr_model=None,
                tts_model_path=f"{tts_model_path}",
                tts_model=None,
                enhancer_model_path=None,
                enhancer_model=None,
                asr_model_type=f"{model_type}",
                asr_model_fuse_bn=False,  # for training from scratch always should be False
                train_ds=None,
                validation_ds=None,
                test_ds=None,
                optim=None,
            )
        )

        asr_cfg = copy.deepcopy(asr_cfg)  # copy not to avoid original config
        OmegaConf.set_struct(asr_cfg, False)
        for subconfig_path in ["optim", "train_ds", "validation_ds", "test_ds"]:
            if subconfig_path in asr_cfg:
                cfg[subconfig_path] = asr_cfg.pop(subconfig_path)
        cfg.asr_model = asr_cfg
        return cls(cfg=cfg, trainer=trainer)

    @classmethod
    def from_pretrained_models(
        cls,
        asr_model_path: Union[str, Path],
        tts_model_path: Union[str, Path],
        asr_model_fuse_bn: bool = False,
        cfg: Optional[DictConfig] = None,
        trainer: Optional[Trainer] = None,
    ):
        if cfg is None:
            cfg = DictConfig(
                dict(
                    asr_model_path=f"{asr_model_path}",
                    asr_model=None,
                    tts_model_path=f"{tts_model_path}",
                    tts_model=None,
                    enhancer_model_path=None,
                    enhancer_model=None,
                    asr_model_type=None,
                    asr_model_fuse_bn=asr_model_fuse_bn,
                    train_ds=None,
                    validation_ds=None,
                    test_ds=None,
                    optim=None,
                )
            )
        else:
            cfg.tts_model_path = f"{tts_model_path}"
            cfg.asr_model_path = f"{asr_model_path}"
        return ASRWithTTSModel(cfg, trainer=trainer)

    def __setattr__(self, name, value):
        if name == "_current_fx_name" and self._full_init_guard:
            # needed to call *_step on asr_model
            # FixMe: use on_* methods
            self.asr_model._current_fx_name = value
        return super().__setattr__(name, value)

    def setup_optimization(
        self, optim_config: Optional[Union[DictConfig, Dict]] = None, optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.tts_model.freeze()
        optimizer, scheduler = super().setup_optimization(optim_config=optim_config, optim_kwargs=optim_kwargs)
        # set asr model optimizer/scheduler to allow training_step on asr_model
        self.asr_model._optimizer = optimizer
        self.asr_model._scheduler = scheduler
        return optimizer, scheduler

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        return self.asr_model.setup_validation_data(val_data_config)

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.asr_model.multi_validation_epoch_end(outputs=outputs, dataloader_idx=dataloader_idx)

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.asr_model.multi_test_epoch_end(outputs=outputs, dataloader_idx=dataloader_idx)

    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[str]:
        return self.asr_model.transcribe(paths2audio_files=paths2audio_files, batch_size=batch_size)

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self.asr_model.setup_multiple_validation_data(val_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self.asr_model.setup_test_data(test_data_config)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        return self.asr_model.setup_multiple_test_data(test_data_config)

    def save_asr_model_to(self, save_path: str):
        return self.asr_model.save_to(save_path=save_path)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.asr_model.validation_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def validation_epoch_end(
        self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        return self.asr_model.validation_epoch_end(outputs=outputs)

    def test_epoch_end(
        self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        return self.asr_model.test_epoch_end(outputs=outputs)

    def val_dataloader(self):
        return self.asr_model.val_dataloader()

    def unfreeze(self) -> None:
        super().unfreeze()
        self.tts_model.freeze()  # tts model should be always frozen

    def on_fit_start(self):
        self.asr_model.on_fit_start()
        self.tts_model.freeze()

    def train(self, mode: bool = True):
        super().train(mode)
        self.tts_model.eval()
        return self

    def _get_tts_spectrogram(
        self, tts_texts: torch.Tensor, speakers: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            spectrogram, spectrogram_len, *_ = self.tts_model(text=tts_texts, durs=None, pitch=None, speaker=speakers)
            # ToDo: enhancer

            spectrogram, *_ = normalize_batch(spectrogram, spectrogram_len, self.asr_model.cfg.preprocessor.normalize)
            return spectrogram, spectrogram_len

    def _get_batch_spect(self, batch: Union[TextToTextBatch, TextOrAudioToTextBatch, tuple]):
        if isinstance(batch, TextToTextBatch):
            spectrogram, spectrogram_len = self._get_tts_spectrogram(batch.tts_texts, batch.speakers)
            transcript = batch.transcripts
            transcript_len = batch.transcript_length
        elif isinstance(batch, TextOrAudioToTextBatch):
            tts_spectrogram, tts_spectrogram_len = self.get_tts_spectrogram(batch.tts_texts, batch.speakers)
            asr_spectrogram, asr_spectrogram_len = self.preprocessor(
                input_signal=batch.audio_signal, length=batch.a_sig_length,
            )

            spectrogram = pad_sequence(
                [
                    x.squeeze(0)
                    for x in itertools.chain(
                        torch.tensor_split(tts_spectrogram.transpose(1, 2), tts_spectrogram.size(0)),
                        torch.tensor_split(asr_spectrogram.transpose(1, 2), asr_spectrogram.size(0)),
                    )
                ],
                batch_first=True,
                padding_value=0.0,
            ).transpose(1, 2)
            spectrogram_len = torch.cat([tts_spectrogram_len, asr_spectrogram_len], dim=0)

            transcript = batch.transcripts
            transcript_len = batch.transcript_length
        else:
            audio_signal, audio_signal_len, transcript, transcript_len = batch
            spectrogram, spectrogram_len = self.preprocessor(input_signal=audio_signal, length=audio_signal_len,)
        spectrogram = clean_spectrogram_batch(spectrogram, spectrogram_len)
        return spectrogram.detach(), spectrogram_len.detach(), transcript, transcript_len

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if train_data_config is None:
            logging.warning("No training data")
            return

        self._update_dataset_config(dataset_name='train', config=train_data_config)
        if train_data_config.get("text_data") is not None:
            tts_dataset = self._setup_text_dataset_from_config(train_data_config)
        else:
            tts_dataset = None
        asr_dataset = self.asr_model._setup_dataset_from_config(train_data_config)
        if tts_dataset and asr_dataset:
            raise NotImplementedError
            # dataset = ConcatDataset(tts_dataset, asr_dataset)  # FixMe: implementation
        else:
            dataset = tts_dataset or asr_dataset
        if dataset is None:
            return None
        collate_fn = dataset.collate_fn
        self._train_dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=train_data_config['batch_size'],
            collate_fn=collate_fn,
            drop_last=train_data_config.get('drop_last', False),
            shuffle=train_data_config.get('shuffle', True),
            num_workers=train_data_config.get('num_workers', 0),
            pin_memory=train_data_config.get('pin_memory', False),
        )

    def _setup_text_dataset_from_config(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        text_data_config = train_data_config.text_data
        textonly_ds = TextToTextDataset(
            manifest_filepath=text_data_config.manifest_filepath,
            speakers_filepath=text_data_config.speakers_filepath,
            asr_tokenizer=self.asr_model.tokenizer,
            asr_use_start_end_token=train_data_config.use_start_end_token,
            tts_text_normalizer=self.tts_model.normalizer,
            tts_text_normalizer_call_kwargs=self.tts_model.text_normalizer_call_kwargs,
            tts_parser=self.tts_model.parser,
            tts_text_pad_id=self.tts_model.vocab.pad,
            min_words=text_data_config.get("min_words", 1),
            max_words=text_data_config.get("max_words", 1_000_000),  # 45 - recommended value, ~16.7 sec
            tokenizer_workers=text_data_config.get('tokenizer_workers', 1),
        )
        return textonly_ds

    def training_step(self, batch: Union[TextOrAudioToTextBatch, TextToTextBatch, DALIOutputs, tuple], batch_nb: int):
        assert not self.tts_model.training
        if isinstance(batch, DALIOutputs):
            return self.asr_model.training_step(batch=batch, batch_nb=batch_nb)
        with torch.no_grad():
            spectrogram, spectrogram_len, transcript, transcript_len = self._get_batch_spect(batch)
        # FixMe: maybe support precomputed without DALIOutputs
        return self.asr_model.training_step(
            batch=DALIOutputs(
                dict(
                    processed_signal=spectrogram,
                    processed_signal_len=spectrogram_len,
                    transcript=transcript,
                    transcript_len=transcript_len,
                )
            ),
            batch_nb=batch_nb,
        )
