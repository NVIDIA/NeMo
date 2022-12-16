import copy
import itertools
import tempfile
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.data.text_to_text import TextOrAudioToTextBatch, TextToTextBatch, TextToTextDataset
from nemo.collections.asr.models import ASRModel, EncDecCTCModelBPE, EncDecRNNTBPEModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.asr.parts.submodules.batchnorm import FusedBatchNorm1d
from nemo.collections.tts.models import FastPitchModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.enum import PrettyStrEnum


def print_appstate():
    # FixMe: remove
    app_state = AppState()
    print("=" * 50)
    print("App State")
    print(app_state.nemo_file_folder)
    print(app_state.model_restore_path)
    print(app_state._tmpdir_name)
    print(app_state._is_model_being_restored)
    print(app_state._all_model_restore_paths)
    print(app_state._model_guid_map)
    print("=" * 50)


@contextmanager
def _preserve_nemo_file_folder():
    """
    Preserve singleton AppState when combining 2 nemo models
    """
    app_state = AppState()
    nemo_file_folder = app_state.nemo_file_folder
    try:
        yield
    finally:
        # if nemo_file_folder:
        AppState().nemo_file_folder = nemo_file_folder


def clean_spectrogram(spectrogram: torch.Tensor, spectrogram_len: torch.Tensor, fill_value=0.0) -> torch.Tensor:
    device = spectrogram.device
    batch_size, num_channels, max_len = spectrogram.shape
    mask = torch.arange(max_len, device=device)[None, :] >= spectrogram_len[:, None]
    mask = mask.unsqueeze(1).expand_as(spectrogram)
    return spectrogram.masked_fill(mask, fill_value)


def get_module_by_name(module: nn.Module, full_layer_name: str) -> nn.Module:
    names = full_layer_name.split(sep='.')
    return reduce(getattr, names, module)


def replace_bn_with_fused_bn(module: nn.Module, full_layer_name: str) -> bool:
    bn = get_module_by_name(module, full_layer_name)
    if not isinstance(bn, nn.BatchNorm1d):
        logging.warning(f"Expected BatchNorm1d, found {bn}")
        return False
    fused_bn = FusedBatchNorm1d.from_batchnorm(bn)
    parent_name, norm_name = full_layer_name.rsplit(".", maxsplit=1)
    setattr(get_module_by_name(module, parent_name), norm_name, fused_bn)
    return True


def fuse_bn_in_encoder(model: nn.Module):
    try:
        pass
    except AttributeError:
        raise ValueError(f"No encoder found in asr model {model}")
    replaced_in_model = False
    for name, module in model.named_modules():
        if "batch_norm" in name.split("."):
            replaced = replace_bn_with_fused_bn(model, name)
            replaced_in_model = replaced_in_model or replaced
    if replaced_in_model:
        logging.info("Successfully replace BatchNorm with FusedBatchNorm")
    else:
        logging.warning(f"No BatchNorm to replace with FusedBatchNorm in model {model}")


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
            if self.value == self.RNNT_BPE:
                return EncDecRNNTBPEModel
            if self.value == self.CTC_BPE:
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

        with _preserve_nemo_file_folder():
            tts_model_path = self.register_artifact("tts_model_path", cfg.get("tts_model_path"))
            tts_model = FastPitchModel.restore_from(tts_model_path, map_location=torch.device("cpu"))

        if "asr_model_path" in cfg and cfg.get("asr_model_path") is not None:
            with _preserve_nemo_file_folder():
                asr_model_path = self.register_artifact("asr_model_path", cfg.get("asr_model_path"))
                asr_model = ASRModel.restore_from(asr_model_path, map_location=torch.device("cpu"))
                self.asr_model_type = self.ASRModelTypes.from_asr_model(asr_model)
                OmegaConf.set_struct(self.cfg, False)
                self.cfg.asr_model_type = str(self.asr_model_type)
                OmegaConf.set_struct(self.cfg, True)
                # get optimizer config from ASR model
                if optim_cfg is None:
                    optim_cfg = asr_model.cfg.get("optim", None)
        else:
            model_type_str = cfg.get("asr_model_type")
            self.asr_model_type = self.ASRModelTypes(model_type_str)  # convert to enum
            # instantiate asr model from config
            with _preserve_nemo_file_folder():
                asr_model = self.asr_model_type.get_asr_cls()(cfg)
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    save_path = str(Path(tmp_dir_name) / "asr_model.nemo")
                    asr_model.save_to(save_path)
                    asr_model_path = self.register_artifact("asr_model_path", cfg.get("asr_model_path"))
                    asr_model = ASRModel.restore_from(asr_model_path, map_location=torch.device("cpu"))

        if cfg.get("enhancer_model_path", None) is not None:
            # ToDo: add enhancer support after https://github.com/NVIDIA/NeMo/pull/5565
            raise NotImplementedError

        # replace BatchNorm with FusedBatchNorm
        if cfg.get("asr_model_fuse_bn"):
            logging.info("Trying to replace BatchNorm with Fused BatchNorm")
            if not hasattr(asr_model, "encoder"):
                raise NotImplementedError("No encoder found in ASR Model, replacement not supported")
            if not isinstance(asr_model.encoder, ConformerEncoder):
                raise NotImplementedError(f"Unsupported encoder type: {type(asr_model.encoder)}")
            fuse_bn_in_encoder(asr_model.encoder)
            asr_model.cfg.encoder.conv_norm_type = "fused_batch_norm"

        self.asr_model = asr_model
        self.tts_model = tts_model
        self.tts_model.freeze()
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
        cls, cfg: DictConfig, model_type: Union[str, ASRModelTypes], tts_model: FastPitchModel, trainer: Trainer = None
    ):
        raise NotImplementedError()

    @classmethod
    def from_pretrained_models(
        cls,
        asr_model_path: str,
        tts_model_path: str,
        cfg: DictConfig = None,
        trainer: Optional[Trainer] = None,
    ):
        if cfg is None:
            cfg = DictConfig(dict())
        cfg.tts_model_path = tts_model_path
        cfg.asr_model_path = asr_model_path
        return ASRWithTTSModel(cfg, trainer=trainer)

    def __setattr__(self, name, value):
        if name == "_current_fx_name" and self._full_init_guard:
            # needed to call *_step on asr_model
            self.asr_model._current_fx_name = value
        return super().__setattr__(name, value)

    def setup_optimization(
        self,
        optim_config: Optional[Union[DictConfig, Dict]] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.tts_model.freeze()
        optimizer, scheduler = super().setup_optimization(optim_config=optim_config, optim_kwargs=optim_kwargs)
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

    def save_tts_model_to(self, save_path: str):
        return self.tts_model.save_to(save_path=save_path)

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
                input_signal=batch.audio_signal,
                length=batch.a_sig_length,
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
            spectrogram, spectrogram_len = self.preprocessor(
                input_signal=audio_signal,
                length=audio_signal_len,
            )
        spectrogram = clean_spectrogram(spectrogram, spectrogram_len)
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
            # dataset = ConcatDataset(tts_dataset, asr_dataset)  # fixme
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
        # self.asr_model._train_dl = self._train_dl  # for scheduler

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
