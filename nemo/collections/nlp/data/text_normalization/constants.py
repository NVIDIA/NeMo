# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

DECODE_CTX_SIZE = 3  # the size of the input context to be provided to the DuplexDecoderModel
LABEL_PAD_TOKEN_ID = -100

# Split names
TRAIN, DEV, TEST = 'train', 'dev', 'test'
SPLIT_NAMES = [TRAIN, DEV, TEST]

# Languages
ENGLISH = 'en'
RUSSIAN = 'ru'
GERMAN = 'de'
MULTILINGUAL = 'multilingual'
SUPPORTED_LANGS = [ENGLISH, RUSSIAN, GERMAN, MULTILINGUAL]

# Task Prefixes
ITN_TASK = 0
TN_TASK = 1
ITN_PREFIX = str(ITN_TASK)
TN_PREFIX = str(TN_TASK)

# Tagger Labels Prefixes
B_PREFIX = 'B-'  # Denote beginning
I_PREFIX = 'I-'  # Denote middle
TAGGER_LABELS_PREFIXES = [B_PREFIX, I_PREFIX]

# Modes
TN_MODE = 'tn'
ITN_MODE = 'itn'
JOINT_MODE = 'joint'
MODES = [TN_MODE, ITN_MODE, JOINT_MODE]
TASK_ID_TO_MODE = {ITN_TASK: ITN_MODE, TN_TASK: TN_MODE}
MODE_TO_TASK_ID = {v: k for k, v in TASK_ID_TO_MODE.items()}

# Instance Directions
INST_BACKWARD = 'BACKWARD'
INST_FORWARD = 'FORWARD'
INST_DIRECTIONS = [INST_BACKWARD, INST_FORWARD]
DIRECTIONS_TO_ID = {INST_BACKWARD: ITN_TASK, INST_FORWARD: TN_TASK}
DIRECTIONS_ID_TO_NAME = {ITN_TASK: INST_BACKWARD, TN_TASK: INST_FORWARD}
DIRECTIONS_TO_MODE = {ITN_MODE: INST_BACKWARD, TN_MODE: INST_FORWARD}

# TAGS
SAME_TAG = 'SAME'  # Tag indicates that a token can be kept the same without any further transformation
TASK_TAG = 'TASK'  # Tag indicates that a token belongs to a task prefix (the prefix indicates whether the current task is TN or ITN)
PUNCT_TAG = 'PUNCT'  # Tag indicates that a token is a punctuation
TRANSFORM_TAG = 'TRANSFORM'  # Tag indicates that a token needs to be transformed by the decoder
ALL_TAGS = [TASK_TAG, SAME_TAG, TRANSFORM_TAG]

# ALL_TAG_LABELS
ALL_TAG_LABELS = []
for prefix in TAGGER_LABELS_PREFIXES:
    for tag in ALL_TAGS:
        ALL_TAG_LABELS.append(prefix + tag)

ALL_TAG_LABELS.sort()
LABEL_IDS = {l: idx for idx, l in enumerate(ALL_TAG_LABELS)}

# Special Words
SIL_WORD = 'sil'
SELF_WORD = '<self>'
SPECIAL_WORDS = [SIL_WORD, SELF_WORD]

# IDs for special tokens for encoding inputs of the decoder models
EXTRA_ID_0 = '<extra_id_0>'
EXTRA_ID_1 = '<extra_id_1>'


EN_GREEK_TO_SPOKEN = {
    'Τ': 'tau',
    'Ο': 'omicron',
    'Δ': 'delta',
    'Η': 'eta',
    'Κ': 'kappa',
    'Ι': 'iota',
    'Θ': 'theta',
    'Α': 'alpha',
    'Σ': 'sigma',
    'Υ': 'upsilon',
    'Μ': 'mu',
    'Χ': 'chi',
    'Π': 'pi',
    'Ν': 'nu',
    'Λ': 'lambda',
    'Γ': 'gamma',
    'Β': 'beta',
    'Ρ': 'rho',
    'τ': 'tau',
    'υ': 'upsilon',
    'φ': 'phi',
    'α': 'alpha',
    'λ': 'lambda',
    'ι': 'iota',
    'ς': 'sigma',
    'ο': 'omicron',
    'σ': 'sigma',
    'η': 'eta',
    'π': 'pi',
    'ν': 'nu',
    'γ': 'gamma',
    'κ': 'kappa',
    'ε': 'epsilon',
    'β': 'beta',
    'ρ': 'rho',
    'ω': 'omega',
    'χ': 'chi',
}
