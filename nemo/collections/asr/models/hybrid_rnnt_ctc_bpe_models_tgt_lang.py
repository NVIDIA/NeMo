# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Temporary compatibility file to bridge the old model with the new prompt-based model.
This file is needed to load old checkpoints and convert them to the new format.
"""

from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import EncDecHybridRNNTCTCBPEModelWithPrompt

# Global map of language codes to indices that existed in the old model
GLOBAL_LANG_MAP = {
    'en-US': 0,
    'en-GB': 1,
    'es-ES': 2,
    'es-US': 3,
    'zh-CN': 4,
    'zh-TW': 5,
    'hi-IN': 6,
    'ar-AR': 7,
    'fr-FR': 8,
    'de-DE': 9,
    'ja-JP': 10,
    'ru-RU': 11,
    'pt-BR': 12,
    'pt-PT': 13,
    'ko-KR': 14,
    'it-IT': 15,
    'nl-NL': 16,
    'pl-PL': 17,
    'tr-TR': 18,
    'uk-UA': 19,
    'ro-RO': 20,
    'el-GR': 21,
    'cs-CZ': 22,
    'hu-HU': 23,
    'sv-SE': 24,
    'da-DK': 25,
    'fi-FI': 26,
    'no-NO': 27,
    'sk-SK': 28,
    'hr-HR': 29,
    'bg-BG': 30,
    'lt-LT': 31,
    'th-TH': 32,
    'vi-VN': 33,
    'id-ID': 34,
    'ms-MY': 35,
    'bn-IN': 36,
    'ur-PK': 37,
    'fa-IR': 38,
    'ta-IN': 39,
    'te-IN': 40,
    'mr-IN': 41,
    'gu-IN': 42,
    'kn-IN': 43,
    'ml-IN': 44,
    'si-LK': 45,
    'ne-NP': 46,
    'km-KH': 47,
    'sw-KE': 48,
    'am-ET': 49,
    'ha-NG': 50,
    'zu-ZA': 51,
    'yo-NG': 52,
    'ig-NG': 53,
    'af-ZA': 54,
    'rw-RW': 55,
    'so-SO': 56,
    'ny-MW': 57,
    'ln-CD': 58,
    'or-KE': 59,
    'he-IL': 64,
    'ku-TR': 65,
    'az-AZ': 66,
    'ka-GE': 67,
    'hy-AM': 68,
    'uz-UZ': 69,
    'tg-TJ': 70,
    'ky-KG': 71,
    'qu-PE': 80,
    'ay-BO': 81,
    'gn-PY': 82,
    'nah-MX': 83,
    'mi-NZ': 96,
    'haw-US': 97,
    'sm-WS': 98,
    'to-TO': 99,
}


# For backward compatibility
class EncDecHybridRNNTCTCBPEModelTgtLangID(EncDecHybridRNNTCTCBPEModelWithPrompt):
    """Compatibility class for loading old checkpoints"""

    def __init__(self, cfg, trainer=None):
        # Add prompt_dictionary to cfg if not exists
        if cfg is not None:
            if not hasattr(cfg, 'model_defaults'):
                with open_dict(cfg):
                    cfg.model_defaults = OmegaConf.create({})

            if not hasattr(cfg.model_defaults, 'prompt_dictionary'):
                with open_dict(cfg.model_defaults):
                    cfg.model_defaults.prompt_dictionary = GLOBAL_LANG_MAP

            # Handle missing model section
            if not hasattr(cfg, 'model'):
                with open_dict(cfg):
                    cfg.model = OmegaConf.create({})
                    cfg.model.subsampling_factor = 8

        super().__init__(cfg, trainer)

        # Save num_langs for backward compatibility
        self.num_langs = self.num_prompts

    def load_state_dict(self, state_dict, strict=True):
        # Map the old keys to new keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("lang_kernal"):
                new_key = key.replace("lang_kernal", "lang_kernel")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # Call the parent's load_state_dict with the fixed dictionary
        return super().load_state_dict(new_state_dict, strict)

    # Add compatibility methods
    def convert_to_prompt_model(self, output_path):
        """
        Convert this model to a proper prompt-based model and save to a new checkpoint.

        Args:
            output_path: Path to save the new model checkpoint

        Returns:
            Path to the new model checkpoint
        """
        # The model is already using the prompt-based implementation
        # Just need to save it with the new class
        self.save_to(output_path)
        return output_path


# Alias for backwards compatibility
EncDecHybridRNNTCTCBPEModelWithTgtLang = EncDecHybridRNNTCTCBPEModelTgtLangID
