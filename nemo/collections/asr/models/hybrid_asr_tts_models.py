import copy
import itertools
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.data.text_to_text import TextOrAudioToTextBatch, TextToTextBatch, TextToTextDataset
from nemo.collections.asr.models import ASRModel, EncDecCTCModelBPE, EncDecRNNTBPEModel
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.tts.models import FastPitchModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils
from nemo.utils.app_state import AppState
from nemo.utils.enum import PrettyStrEnum


def clean_spectrogram(spectrogram: torch.Tensor, spectrogram_len: torch.Tensor, fill_value=0.0) -> torch.Tensor:
    device = spectrogram.device
    batch_size, num_channels, max_len = spectrogram.shape
    mask = torch.arange(max_len, device=device)[None, :] >= spectrogram_len[:, None]
    mask = mask.unsqueeze(1).expand_as(spectrogram)
    return spectrogram.masked_fill(mask, fill_value)


class ASRWithTTSModel(ASRModel):
    asr_model: Union[EncDecRNNTBPEModel, EncDecCTCModelBPE]

    class ASRModelTypes(PrettyStrEnum):
        RNNT_BPE = "rnnt_bpe"
        CTC_BPE = "ctc_bpe"

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        super().__init__(cfg, trainer=trainer)
        asr_config = copy.deepcopy(cfg)
        model_type_str = asr_config.pop("asr_model_type")
        tts_config = asr_config.pop("tts_model")

        model_type = self.ASRModelTypes(model_type_str)  # convert to enum

        if model_type == self.ASRModelTypes.CTC_BPE:
            self.asr_model = EncDecCTCModelBPE(cfg=cfg)
        elif model_type == self.ASRModelTypes.RNNT_BPE:
            self.asr_model = EncDecRNNTBPEModel(cfg=cfg)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.tts_model = FastPitchModel(cfg=cfg, trainer=None)
        self.tts_model.freeze()

    @classmethod
    def from_asr_config(
        cls, cfg: DictConfig, model_type: Union[str, ASRModelTypes], tts_model: FastPitchModel, trainer: Trainer = None
    ):
        pass

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
    ):
        return self.asr_model.change_vocabulary(
            new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type=new_tokenizer_type, decoding_cfg=decoding_cfg
        )

    def change_decoding_strategy(self, decoding_cfg: DictConfig):
        return self.asr_model.change_decoding_strategy(decoding_cfg=decoding_cfg)

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
            if isinstance(batch, DALIOutputs):
                raise NotImplementedError  # FixMe
            else:
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

        if train_data_config.get("textonly_manifest_filepath") is not None:
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
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_text_dataset_from_config(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        assert "textonly_manifest_filepath" in train_data_config
        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        textonly_ds = TextToTextDataset(
            manifest_filepath=train_data_config.textonly_manifest_filepath,
            speakers_filepath=train_data_config.speakers_filepath,
            asr_tokenizer=self.asr_tokenizer,
            asr_use_start_end_token=train_data_config.use_start_end_token,
            tts_text_normalizer=self.tts_normalizer,
            tts_text_normalizer_call_kwargs=self.tts_text_normalizer_call_kwargs,
            tts_parser=self.tts_parser,
            tts_text_pad_id=self.tts_text_pad,
            min_words=train_data_config.min_words,
            max_words=train_data_config.max_words,
            tokenizer_workers=train_data_config.dl_workers.get('num_workers', 1),
        )
        return textonly_ds

    # @classmethod
    # def restore_from_separate_models(
    #     cls,
    #     asr_restore_path: str,
    #     tts_restore_path: str,
    #     asr_override_config_path: Optional[Union[OmegaConf, str]] = None,
    #     tts_override_config_path: Optional[Union[OmegaConf, str]] = None,
    #     asr_strict: bool = True,
    #     tts_strict: bool = True,
    #     map_location: Optional[torch.device] = None,
    #     trainer: Optional[Trainer] = None,
    # ):
    #     asr_model = ASRModel.restore_from(restore_path=asr_restore_path, override_config_path=asr_override_config_path, map_location=map_location, strict=asr_strict)
    #     tts_model = FastPitchModel.restore_from(restore_path=tts_restore_path, override_config_path=tts_override_config_path, map_location=map_location, strict=tts_strict)

    # def from_models(cls, asr_model: ASRModel, tts_model: FastPitchModel, asr_model_type: Union[str,
