from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn import Linear

import nemo.collections.asr as nemo_asr

from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import \
    PunctuationCapitalizationLexicalAudioTrainDataConfig, PunctuationCapitalizationLexicalAudioEvalDataConfig
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_lexical_audio_dataset import \
    PunctuationCapitalizationLexicalAudioDataset
from nemo.collections.nlp.metrics import ClassificationReport
from nemo.collections.nlp.models import PunctuationCapitalizationModel
from nemo.collections.nlp.modules.common.transformer import TransformerDecoder
from nemo.utils import logging

__all__ = ['PunctuationCapitalizationLexicalAudioModel']


class PunctuationCapitalizationLexicalAudioModel(PunctuationCapitalizationModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg, trainer)
        self.audio_encoder = nemo_asr.models.ASRModel.from_pretrained(cfg.pretrained_audio_encoder)  # Only CTC models?
        del self.audio_encoder.decoder
        self.fusion = TransformerDecoder(num_layers=4, hidden_size=768, inner_size=2048, num_attention_heads=4)
        self.audio_proj = Linear(256, 768)

    def _make_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        punct_logits, capit_logits = self(
            input_ids=batch['input_ids'], token_type_ids=batch['segment_ids'], attention_mask=batch['input_mask'],
            features=batch['features'], features_length=batch['features_length']
        )

        punct_loss = self.loss(logits=punct_logits, labels=batch['punct_labels'], loss_mask=batch['loss_mask'])
        capit_loss = self.loss(logits=capit_logits, labels=batch['capit_labels'], loss_mask=batch['loss_mask'])
        loss = self.agg_loss(loss_1=punct_loss, loss_2=capit_loss)
        return loss, punct_logits, capit_logits

    def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
            features: torch.Tensor = None, features_length: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cfg.tokenizer.get('library', '') == 'megatron':
            lexical_hidden_states, _ = self.bert_model(input_ids, attention_mask, tokentype_ids=token_type_ids,
                                                       lm_labels=None)
        else:
            lexical_hidden_states = self.bert_model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
        processed_signal, processed_signal_length = self.audio_encoder.preprocessor(
            input_signal=features, length=features_length,
        )

        if self.audio_encoder.spec_augmentation is not None and self.training:
            processed_signal = self.audio_encoder.spec_augmentation(input_spec=processed_signal,
                                                                    length=processed_signal_length)

        audio_hidden_states, audio_hidden_states_length = self.audio_encoder.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length)
        audio_hidden_states = audio_hidden_states.permute(0, 2, 1)
        audio_hidden_states = self.audio_proj(audio_hidden_states)

        fused = self.fusion(lexical_hidden_states, attention_mask,
                            audio_hidden_states,
                            self.audio_encoder.encoder.make_pad_mask(audio_hidden_states_length.max(),
                                                                     audio_hidden_states_length))
        punct_logits = self.punct_classifier(hidden_states=fused)
        capit_logits = self.capit_classifier(hidden_states=fused)

        return punct_logits, capit_logits

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return

    def setup_training_data(self, train_data_config: Optional[Union[Dict[str, Any], DictConfig]] = None) -> None:
        """
                Sets up training data: creates dataset and sets data loader. If parameter ``train_data_config`` is not
                provided, then :ref:`config<model-config-label>` section ``train_ds`` will be used.

                Args:
                    train_data_config (:obj:`Union[Dict[str, Any], DictConfig]`, `optional`): a dictionary that should contain
                        only fields present in :ref:`data config<data-config-label>`.
                        If some of the fields are missing, then they will be set according to
                        :ref:`data config<data-config-label>` defaults. If ``train_data_config`` parameter is not set, then
                        ``train_ds`` item of model config is used. Here model config is a configuration used for model
                        instantiation.
        """
        if train_data_config is not None:
            train_data_config = OmegaConf.create(train_data_config)
            train_data_config = OmegaConf.merge(
                OmegaConf.structured(PunctuationCapitalizationLexicalAudioTrainDataConfig), train_data_config
            )
        if train_data_config is None:
            train_data_config = self._cfg.train_ds

        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, train=True)
        self.punct_label_ids = self._train_dl.dataset.punct_label_ids.copy()
        self.capit_label_ids = self._train_dl.dataset.capit_label_ids.copy()
        self.label_ids_are_set = True
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            label_vocab_dir = self._cfg.common_dataset_parameters.label_vocab_dir
            if label_vocab_dir is None:
                punct_label_ids_file, capit_label_ids_file = self._train_dl.dataset.save_labels_and_get_file_paths(
                    self._cfg.class_labels.punct_labels_file, self._cfg.class_labels.capit_labels_file
                )
            else:
                punct_label_ids_file = Path(label_vocab_dir).expanduser() / self._cfg.class_labels.punct_labels_file
                capit_label_ids_file = Path(label_vocab_dir).expanduser() / self._cfg.class_labels.capit_labels_file
            self.register_artifact('class_labels.punct_labels_file', str(punct_label_ids_file))
            self.register_artifact('class_labels.capit_labels_file', str(capit_label_ids_file))
        # audio_config = self.audio_encoder.config

    def setup_validation_data(self, val_data_config: Optional[Union[Dict[str, Any], DictConfig]] = None) -> None:
        """
        Sets up validation data: creates dataset and sets data loader. If parameter ``val_data_config`` is not
        provided, then ``validation_ds`` :ref:`config <model-config-label>` section will be used. Here model config is
        a configuration used for model instantiation.

        Args:
            val_data_config (:obj:`Union[Dict[str, Any], DictConfig]`, `optional`): a dictionary that should contain
                only fields present in data config :ref:`description<data-config-label>`.
                If some of the fields are missing, then they will be set according to data config
                :ref:`description<data-config-label>` defaults. If ``val_data_config`` parameter is not set, then
                ``validation_ds`` item of model config is used. Here model config is a configuration used for model
                instantiation.
        """
        if val_data_config is not None:
            val_data_config = OmegaConf.create(val_data_config)
            val_data_config = OmegaConf.merge(
                OmegaConf.structured(PunctuationCapitalizationLexicalAudioEvalDataConfig), val_data_config
            )
        if self.metrics is None:
            self._setup_metrics_dictionary()
        if val_data_config is None:
            val_data_config = self._cfg.validation_ds

        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, train=False)
        loss_kw, punct_kw, capit_kw = self._get_eval_metrics_kwargs()
        self.metrics['val']['loss'].append(GlobalAverageLossMetric(**loss_kw))
        self.metrics['val']['punct_class_report'].append(ClassificationReport(**punct_kw))
        self.metrics['val']['capit_class_report'].append(ClassificationReport(**capit_kw))

    def setup_test_data(self, test_data_config: Optional[Union[Dict[str, Any], DictConfig]] = None) -> None:
        """
        Sets up test data: creates dataset and sets data loader. If parameter ``test_data_config`` is not
        provided, then ``test_ds`` config section will be used. See more about in data config
        :ref:`description <data-config-label>` and model config :ref:`description<model-config-label>`.

        Args:
            test_data_config (:obj:`Union[Dict[str, Any], DictConfig]`, `optional`): a dictionary that should contain
                only fields present in data config :ref:`description<data-config-label>`.
                If some of the fields are missing, then they will be set according to data config
                :ref:`description <data-config-label>` defaults. If ``test_data_config`` parameter is not set, then
                ``test_ds`` item of :ref:`model config <model-config-label>` is used. Here model config is a
                configuration used for model instantiation.
        """
        if test_data_config is not None:
            test_data_config = OmegaConf.create(test_data_config)
            test_data_config = OmegaConf.merge(
                OmegaConf.structured(PunctuationCapitalizationLexicalAudioEvalDataConfig), test_data_config
            )
        if self.metrics is None:
            self._setup_metrics_dictionary()
        if test_data_config is None:
            test_data_config = self._cfg.test_ds
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, train=False)
        loss_kw, punct_kw, capit_kw = self._get_eval_metrics_kwargs()
        self.metrics['test']['loss'].append(GlobalAverageLossMetric(**loss_kw))
        self.metrics['test']['punct_class_report'].append(ClassificationReport(**punct_kw))
        self.metrics['test']['capit_class_report'].append(ClassificationReport(**capit_kw))

    def _setup_dataloader_from_config(self, cfg: DictConfig, train: bool) -> torch.utils.data.DataLoader:
        """
        Creates dataset and data loader according to config ``cfg``. If ``train=False`` and attributes
        ``punct_label_ids`` and ``capit_label_ids`` are not set, then this method sets the attributes and registers
        label artifacts.

        Args:
            cfg (:obj:`DictConfig`): a config which follows dataclass
                :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.PunctuationCapitalizationEvalDataConfig`
                Note that list ``ds_item`` is not supported because list ``ds_item`` is unpacked by NeMo core
                instruments
            train (:obj:`bool`): whether train data is set. If ``True``, then label ids are not set in this function
        """
        self._check_label_config_parameters()
        if not self.label_ids_are_set and not train:
            self._set_label_ids()
        if cfg.use_tarred_dataset:
            raise NotImplementedError("P&C lexical audio model currently doesn't support tarred datasets")
            # if cfg.tar_metadata_file is None:
            #     raise ValueError(
            #         f"If parameter `use_tarred_dataset` is `True`, then a field `tar_metadata_file` has to be a path "
            #         f"to tarred dataset metadata file, whereas `None` is given."
            #     )
            # tar_metadata_file = Path(cfg.ds_item) / cfg.tar_metadata_file
            # dataset = BertPunctuationCapitalizationTarredDataset(
            #     metadata_file=tar_metadata_file,
            #     tokenizer=self.tokenizer,
            #     pad_label=self._cfg.common_dataset_parameters.pad_label,
            #     ignore_extra_tokens=self._cfg.common_dataset_parameters.ignore_extra_tokens,
            #     ignore_start_end=self._cfg.common_dataset_parameters.ignore_start_end,
            #     world_size=self.world_size,
            #     global_rank=self.global_rank,
            #     shuffle_n=cfg.tar_shuffle_n,
            #     label_info_save_dir=cfg.label_info_save_dir,
            # )
            # dataset.check_for_label_consistency_with_model_config(
            #     self.punct_label_ids,
            #     self.capit_label_ids,
            #     self._cfg.class_labels,
            #     self._cfg.common_dataset_parameters,
            # )
        else:
            if cfg.text_file is None or cfg.labels_file is None:
                raise ValueError(
                    f"If parameter `use_tarred_dataset` is `False`, then fields `text_file` and `labels_file` in "
                    f"dataset config must not be `None`. Whereas `text_file={cfg.text_file}` and "
                    f"`label_file={cfg.labels_file}`."
                )
            # if cfg.tokens_in_batch is None:
            #     raise ValueError(
            #         f"If `use_tarred_dataset` is `False`, then you need to provide `tokens_in_batch` parameter."
            #     )
            text_file, labels_file = Path(cfg.ds_item) / cfg.text_file, Path(cfg.ds_item) / cfg.labels_file
            if self.label_ids_are_set:
                label_kwargs = {'punct_label_ids': self.punct_label_ids, 'capit_label_ids': self.capit_label_ids}
            else:
                punct_label_vocab_file, capit_label_vocab_file = self._extract_label_vocab_files_from_config()
                label_kwargs = {
                    'punct_label_ids': self._cfg.common_dataset_parameters.punct_label_ids,
                    'capit_label_ids': self._cfg.common_dataset_parameters.capit_label_ids,
                    'punct_label_vocab_file': punct_label_vocab_file,
                    'capit_label_vocab_file': capit_label_vocab_file,
                }
            dataset = PunctuationCapitalizationLexicalAudioDataset(
                tokenizer=self.tokenizer,
                text_file=text_file,
                labels_file=labels_file,
                pad_label=self._cfg.common_dataset_parameters.pad_label,
                **label_kwargs,
                max_seq_length=cfg.max_seq_length,
                ignore_extra_tokens=self._cfg.common_dataset_parameters.ignore_extra_tokens,
                ignore_start_end=self._cfg.common_dataset_parameters.ignore_start_end,
                use_cache=cfg.use_cache,
                num_samples=cfg.num_samples,
                tokens_in_batch=cfg.tokens_in_batch,
                n_jobs=cfg.n_jobs,
                verbose=cfg.verbose,
                get_label_frequencies=cfg.get_label_frequences,
                cache_dir=cfg.cache_dir,
                label_info_save_dir=cfg.label_info_save_dir,
                manifest_filepath=cfg.audio_manifest_filepath,
                sample_rate=cfg.sample_rate
            )
        if cfg.shuffle and cfg.use_tarred_dataset:
            logging.warning(f"Shuffling in dataloader is not supported for tarred dataset.")
            shuffle = False
        else:
            shuffle = cfg.shuffle
        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            persistent_workers=cfg.persistent_workers,
        )
