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

# Task Prefixes
ITN_PREFIX = str(0)
TN_PREFIX = str(1)

# Tagger Labels Prefixes
B_PREFIX = 'B-'  # Denote beginning
I_PREFIX = 'I-'  # Denote middle
TAGGER_LABELS_PREFIXES = [B_PREFIX, I_PREFIX]

# Modes
TN_MODE = 'tn'
ITN_MODE = 'itn'
JOINT_MODE = 'joint'
MODES = [TN_MODE, ITN_MODE, JOINT_MODE]

# Instance Directions
INST_BACKWARD = 'BACKWARD'
INST_FORWARD = 'FORWARD'
INST_DIRECTIONS = [INST_BACKWARD, INST_FORWARD]

# TAGS
SAME_TAG = 'SAME'  # Tag indicates that a token can be kept the same without any further transformation
TASK_TAG = 'TASK'  # Tag indicates that a token belongs to a task prefix (the prefix indicates whether the current task is TN or ITN)
PUNCT_TAG = 'PUNCT'  # Tag indicates that a token is a punctuation
TRANSFORM_TAG = 'TRANSFORM'  # Tag indicates that a token needs to be transformed by the decoder
ALL_TAGS = [TASK_TAG, SAME_TAG, PUNCT_TAG, TRANSFORM_TAG]

# ALL_TAG_LABELS
ALL_TAG_LABELS = []
for prefix in TAGGER_LABELS_PREFIXES:
    for tag in ALL_TAGS:
        ALL_TAG_LABELS.append(prefix + tag)
ALL_TAG_LABELS.sort()

# Special Words
SIL_WORD = 'sil'
SELF_WORD = '<self>'
SPECIAL_WORDS = [SIL_WORD, SELF_WORD]

# Mappings for Greek Letters
GREEK_TO_SPOKEN = {
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
    'Ε': 'epsilon',
    'Χ': 'chi',
    'Π': 'pi',
    'Ν': 'nu',
    'Λ': 'lambda',
    'Γ': 'gamma',
    'Β': 'beta',
    'Ρ': 'rho',
    'τ': 'tau',
    'υ': 'upsilon',
    'μ': 'mu',
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
SPOKEN_TO_GREEK = {v: k for k, v in GREEK_TO_SPOKEN.items()}

# IDs for special tokens for encoding inputs of the decoder models
EXTRA_ID_0 = '<extra_id_0>'
EXTRA_ID_1 = '<extra_id_1>'
