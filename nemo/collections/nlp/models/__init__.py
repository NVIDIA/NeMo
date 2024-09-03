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


from functools import partial
from nemo.collections.common.module_import_proxy import ModuleImportProxy
namespace = globals()
lazy_import = partial(ModuleImportProxy, global_namespace=namespace)

lazy_import("nemo.collections.nlp.models.duplex_text_normalization", "DuplexDecoderModel")
lazy_import("nemo.collections.nlp.models.duplex_text_normalization", "DuplexTaggerModel")
lazy_import("nemo.collections.nlp.models.duplex_text_normalization", "DuplexTextNormalizationModel")
lazy_import("nemo.collections.nlp.models.entity_linking.entity_linking_model", "EntityLinkingModel")
lazy_import("nemo.collections.nlp.models.glue_benchmark.glue_benchmark_model", "GLUEModel")
lazy_import("nemo.collections.nlp.models.information_retrieval", "BertDPRModel")
lazy_import("nemo.collections.nlp.models.information_retrieval", "BertJointIRModel")
lazy_import("nemo.collections.nlp.models.intent_slot_classification", "IntentSlotClassificationModel")
lazy_import("nemo.collections.nlp.models.intent_slot_classification", "MultiLabelIntentSlotClassificationModel")
lazy_import("nemo.collections.nlp.models.language_modeling", "MegatronGPTPromptLearningModel")
lazy_import("nemo.collections.nlp.models.language_modeling.bert_lm_model", "BERTLMModel")
lazy_import("nemo.collections.nlp.models.language_modeling.transformer_lm_model", "TransformerLMModel")
lazy_import("nemo.collections.nlp.models.machine_translation", "MTEncDecModel")
lazy_import("nemo.collections.nlp.models.question_answering.qa_model", "QAModel")
lazy_import("nemo.collections.nlp.models.spellchecking_asr_customization", "SpellcheckingAsrCustomizationModel")
lazy_import("nemo.collections.nlp.models.text2sparql.text2sparql_model", "Text2SparqlModel")
lazy_import("nemo.collections.nlp.models.text_classification", "TextClassificationModel")
lazy_import("nemo.collections.nlp.models.text_normalization_as_tagging", "ThutmoseTaggerModel")
lazy_import("nemo.collections.nlp.models.token_classification", "PunctuationCapitalizationLexicalAudioModel")
lazy_import("nemo.collections.nlp.models.token_classification", "PunctuationCapitalizationModel")
lazy_import("nemo.collections.nlp.models.token_classification", "TokenClassificationModel")
lazy_import("nemo.collections.nlp.models.zero_shot_intent_recognition", "ZeroShotIntentModel")
