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

from nemo.collections.nlp.nm.non_trainables.dialogue_state_tracking.nlg_multiwoz import TemplateNLGMultiWOZNM
from nemo.collections.nlp.nm.non_trainables.dialogue_state_tracking.rule_based_multiwoz_bot import (
    RuleBasedMultiwozBotNM,
)
from nemo.collections.nlp.nm.non_trainables.dialogue_state_tracking.trade_state_update_nm import TradeStateUpdateNM
from nemo.collections.nlp.nm.non_trainables.dialogue_state_tracking.utterance_encoder_nm import UtteranceEncoderNM
