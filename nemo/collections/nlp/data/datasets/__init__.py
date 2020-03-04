# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

from nemo.collections.nlp.data.datasets.glue_benchmark_dataset.glue_benchmark_dataset import GLUEDataset
from nemo.collections.nlp.data.datasets.joint_intent_slot_dataset.joint_intent_slot_dataset import (
    BertJointIntentSlotDataset,
    BertJointIntentSlotInferDataset,
)
from nemo.collections.nlp.data.datasets.lm_bert_dataset import (
    BertPretrainingDataset,
    BertPretrainingPreprocessedDataset,
)
from nemo.collections.nlp.data.datasets.lm_transformer_dataset import LanguageModelingDataset
from nemo.collections.nlp.data.datasets.machine_translation_dataset import TranslationDataset
from nemo.collections.nlp.data.datasets.multiwoz_dataset import *
from nemo.collections.nlp.data.datasets.punctuation_capitalization_dataset import (
    BertPunctuationCapitalizationDataset,
    BertPunctuationCapitalizationInferDataset,
)
from nemo.collections.nlp.data.datasets.qa_squad_dataset.qa_squad_dataset import SquadDataset
from nemo.collections.nlp.data.datasets.text_classification import (
    BertTextClassificationDataset,
    TextClassificationDataDesc,
)
from nemo.collections.nlp.data.datasets.token_classification_dataset import (
    BertTokenClassificationDataset,
    BertTokenClassificationInferDataset,
)
