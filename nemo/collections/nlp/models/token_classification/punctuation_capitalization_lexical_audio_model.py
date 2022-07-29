from typing import Optional, Dict, Tuple, List

import torch
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.common.losses.cross_entropy import CrossEntropyLoss

from nemo.core.classes.mixins import adapter_mixins
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.nn import Linear

import nemo.collections.asr as nemo_asr

from nemo.collections.nlp.models.token_classification.punctuation_capitalization_model import \
    PunctuationCapitalizationModel
from nemo.collections.nlp.modules.common.transformer import TransformerDecoder
from nemo.core.classes.common import PretrainedModelInfo

__all__ = ['PunctuationCapitalizationLexicalAudioModel']


def update_model_config_to_support_adapter(model_cfg):
    with open_dict(model_cfg):
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path

    return model_cfg


class PunctuationCapitalizationLexicalAudioModel(PunctuationCapitalizationModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg, trainer)
        audio_cfg = nemo_asr.models.ASRModel.from_pretrained(cfg.pretrained_audio_encoder, return_config=True)

        if cfg.get('use_adapters', False):
            audio_cfg = update_model_config_to_support_adapter(audio_cfg)

        self.audio_encoder = nemo_asr.models.ASRModel.from_pretrained(cfg.pretrained_audio_encoder,
                                                                      override_config_path=audio_cfg)
        if cfg.get('use_adapters', False):
            with open_dict(cfg):
                cfg.adapter_config.in_features = self.audio_encoder.cfg.encoder.d_model
            self.audio_encoder.add_adapter(name='audio_adapter', cfg=cfg.adapter_config)
            self.audio_encoder.set_enabled_adapters(enabled=True)
            self.audio_encoder.freeze()
            self.audio_encoder.unfreeze_enabled_adapters()

        self.fusion = TransformerDecoder(num_layers=cfg.fusion_num_layers,
                                         hidden_size=self.bert_model(**self.bert_model.input_example()[0]).size()[-1],
                                         inner_size=cfg.fusion_inner_size,
                                         num_attention_heads=cfg.fusion_num_attention_heads)
        self.audio_proj = Linear(self.audio_encoder.cfg.encoder.d_model,
                                 self.bert_model(**self.bert_model.input_example()[0]).size()[-1])

        if cfg.get('freeze_audio_encoder', False):
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            for i in range(cfg.get('frozen_conf_num_layers')):
                self.audio_encoder.add_module(f'conf_encoder_{i}',
                                              ConformerLayer(d_model=cfg.get('frozen_conf_d_model'),
                                                             d_ff=cfg.get('frozen_conf_d_ff')))

        if cfg.get('restore_lexical_encoder_from', None):
            model = PunctuationCapitalizationModel.restore_from(cfg.restore_lexical_encoder_from).to(self.device)
            self.bert_model.load_state_dict(model.bert_model.state_dict())
            del model

        del self.audio_encoder.decoder
        del self.audio_encoder._wer
        del self.audio_encoder.loss

        if cfg.get('use_weighted_loss', False):
            punct_freq = torch.tensor(list(self.train_dataloader().dataset.punct_label_frequencies.values()),
                                      dtype=torch.float)
            punct_weight = 1 - (punct_freq - punct_freq.min()) / punct_freq.max()

            capit_freq = torch.tensor(list(self.train_dataloader().dataset.capit_label_frequencies.values()),
                                      dtype=torch.float)
            capit_weight = 1 - (capit_freq - capit_freq.min()) / capit_freq.max()

            self.loss_punct = CrossEntropyLoss(logits_ndim=3, weight=punct_weight)
            self.loss_capit = CrossEntropyLoss(logits_ndim=3, weight=capit_weight)
        else:
            self.loss_punct = self.loss
            self.loss_capit = self.loss

    def _make_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        punct_logits, capit_logits = self(
            input_ids=batch['input_ids'], token_type_ids=batch['segment_ids'], attention_mask=batch['input_mask'],
            features=batch['features'], features_length=batch['features_length']
        )

        punct_loss = self.loss_punct(logits=punct_logits, labels=batch['punct_labels'], loss_mask=batch['loss_mask'])
        capit_loss = self.loss_capit(logits=capit_logits, labels=batch['capit_labels'], loss_mask=batch['loss_mask'])
        loss = self.agg_loss(loss_1=punct_loss, loss_2=capit_loss)
        return loss, punct_logits, capit_logits

    def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
            features: torch.Tensor = None, features_length: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lexical_hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        if isinstance(lexical_hidden_states, tuple):
            lexical_hidden_states = lexical_hidden_states[0]

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

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []
