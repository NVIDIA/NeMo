from typing import Dict, Optional, Tuple, List
import torch
import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices

import numpy as np
import math

from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

class LhotseSpeechToTextBpeDatasetTgtLangID(torch.utils.data.Dataset):
    """
    Dataset class for speech-to-text with language ID vectors.
    """
    _GLOBAL_LANG_MAP = {
    # Group 1: 
    'en-US': 0,   'en-GB': 1,   
    'es-ES': 2,   'es-US': 3,   # Spanish variants
    'zh-CN': 4,   'zh-TW': 5,   # Chinese variants
    'hi-IN': 6,   'ar-AR': 7,   # Hindi & Arabic
    'fr-FR': 8,   'de-DE': 9,   # French & German
    'ja-JP': 10,  'ru-RU': 11,  # Japanese & Russian
    'pt-BR': 12,  'pt-PT': 13,  # Portuguese variants
    'ko-KR': 14,  'it-IT': 15,  # Korean & Italian

    # Group 2: 
    'nl-NL': 16,  'pl-PL': 17,  
    'tr-TR': 18,  'uk-UA': 19,
    'ro-RO': 20,  'el-GR': 21,
    'cs-CZ': 22,  'hu-HU': 23,
    'sv-SE': 24,  'da-DK': 25,
    'fi-FI': 26,  'no-NO': 27,
    'sk-SK': 28,  'hr-HR': 29,
    'bg-BG': 30,  'lt-LT': 31,

    # Group 3: 
    'th-TH': 32,  'vi-VN': 33,
    'id-ID': 34,  'ms-MY': 35,
    'bn-IN': 36,  'ur-PK': 37,
    'fa-IR': 38,  'ta-IN': 39,
    'te-IN': 40,  'mr-IN': 41,
    'gu-IN': 42,  'kn-IN': 43,
    'ml-IN': 44,  'si-LK': 45,
    'ne-NP': 46,  'km-KH': 47,

    # Group 4: 
    'sw-KE': 48,  'am-ET': 49,
    'ha-NG': 50,  'zu-ZA': 51,
    'yo-NG': 52,  'ig-NG': 53,
    'af-ZA': 54,  'rw-RW': 55,
    'so-SO': 56,  'ny-MW': 57,
    'ln-CD': 58,  'or-KE': 59,


    # Group 5: 
    'he-IL': 64,  'ku-TR': 65,
    'az-AZ': 66,  'ka-GE': 67,
    'hy-AM': 68,  'uz-UZ': 69,
    'tg-TJ': 70,  'ky-KG': 71,

    'qu-PE': 80,  'ay-BO': 81,
    'gn-PY': 82,  'nah-MX': 83,


    # Group 7: 
    'mi-NZ': 96,  'haw-US': 97,
    'sm-WS': 98,  'to-TO': 99}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
            'target_lang_id': NeuralType(('B', 'T', 'D'), LabelsType())
        }
    
    def __init__(self, tokenizer, cfg):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg
        self.num_languages = cfg.get('num_languages', 128)
        self.num_sample_per_mel_frame = cfg.get('num_sample_per_mel_frame', 160)
        self.num_mel_frame_per_asr_frame = cfg.get('num_mel_frame_per_asr_frame', 8)
    
        # Track unique languages seen during training
        # self.seen_languages = set()
        # self.language_to_index = {}
        # self.next_lang_idx = 0

    def _get_language_index(self, language_code: str) -> int:
        """
        Maps language codes to indices .
        """
        if language_code not in self._GLOBAL_LANG_MAP:
            raise ValueError(f"Unknown language code: {language_code}. Supported languages: {list(self._GLOBAL_LANG_MAP.keys())}")
        
        index = self._GLOBAL_LANG_MAP[language_code]
        
        return index
        
        # if language_code not in self.language_to_index:
        #     if self.next_lang_idx >= self.num_languages:
        #         raise ValueError(f"Number of unique languages ({self.next_lang_idx + 1}) exceeds maximum allowed ({self.num_languages})")
        #     self.language_to_index[language_code] = self.next_lang_idx
        #     self.next_lang_idx += 1
        #     self.seen_languages.add(language_code)
        # return self.language_to_index[language_code]

    def lang_to_target_lang(self, cut, num_languages: int, num_sample_per_mel_frame: int, num_mel_frame_per_asr_frame: int):
        """
        Create language target tensor for the sequence.
        Similar to speaker_to_target but for languages.
        """
        # Calculate encoder output length
        encoder_hidden_len = self.get_hidden_length_from_sample_length(
            cut.num_samples,
            num_sample_per_mel_frame,
            num_mel_frame_per_asr_frame
        )
        
        # Initialize language target matrix
        mask = np.zeros((num_languages, encoder_hidden_len))
        

        # Get language index
        lang_id = self._get_language_index(cut.supervisions[0].language)
        # Set the corresponding language ID to 1 for all time steps
        mask[lang_id, :] = 1
            
        return mask

    # this function is taken from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length        
    def get_hidden_length_from_sample_length(self,
        num_samples: int, 
        num_sample_per_mel_frame: int = 160, 
        num_mel_frame_per_asr_frame: int = 8
    ) -> int:
        """ 
        Calculate the hidden length from the given number of samples.

        This function computes the number of frames required for a given number of audio samples,
        considering the number of samples per mel frame and the number of mel frames per ASR frame.

        Parameters:
            num_samples (int): The total number of audio samples.
            num_sample_per_mel_frame (int, optional): The number of samples per mel frame. Default is 160.
            num_mel_frame_per_asr_frame (int, optional): The number of mel frames per ASR frame. Default is 8.

        Returns:
            hidden_length (int): The calculated hidden length in terms of the number of frames.
        """
        mel_frame_count = math.ceil((num_samples + 1) / num_sample_per_mel_frame)
        hidden_length = math.ceil(mel_frame_count / num_mel_frame_per_asr_frame)
        return int(hidden_length)

    
    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        tokens = [torch.as_tensor(self.tokenizer(c.supervisions[0].text, c.supervisions[0].language)) 
                 for c in cuts]
        
        # Create language targets
        lang_targets = [torch.transpose(torch.as_tensor(self.lang_to_target_lang(
                    c,
                    self.num_languages,
                    self.num_sample_per_mel_frame,
                    self.num_mel_frame_per_asr_frame,
                ),
                dtype=torch.float32
            ),
            0,
            1
        ) for c in cuts]
        
        # Create final tensors
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        lang_targets = collate_matrices(lang_targets)
        
        return (
            audio,           # Audio signal
            audio_lens,      # Audio lengths
            tokens,          # Text tokens
            token_lens,      # Token lengths
            lang_targets,    # Language targets
        )

class TokenizerWrapper:
    """
    Provide a unified interface for NeMo Tokenizer, AggregateTokenizer, and (char) Parser.
    """
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        if isinstance(tokenizer, AggregateTokenizer):
            self._impl = self._call_agg_tokenizer
        elif isinstance(tokenizer, TokenizerSpec):
            self._impl = self._call_tokenizer
        else:
            self._impl = self._call_parser
    
    def __call__(self, text: str, lang: Optional[str] = None):
        return self._impl(text, lang)
    
    def _call_agg_tokenizer(self, text: str, lang: Optional[str] = None):
        assert lang is not None, "Expected 'lang' to be set for AggregateTokenizer."
        return self._tokenizer.text_to_ids(text, lang)
    
    def _call_tokenizer(self, text: str, lang: Optional[str] = None):
        return self._tokenizer.text_to_ids(text)
    
    def _call_parser(self, text: str, lang: Optional[str] = None):
        return self._tokenizer(text)
# from typing import Dict, Optional, Tuple, List
# import torch
# import torch.utils.data
# from lhotse.dataset import AudioSamples
# from lhotse.dataset.collation import collate_vectors, collate_matrices

# import numpy as np
# import math

# from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
# from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
# from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

# class LhotseSpeechToTextBpeDatasetTgtLangID(torch.utils.data.Dataset):
#     """
#     Dataset class for speech-to-text with language ID vectors.
#     """
    
#     @property
#     def output_types(self) -> Optional[Dict[str, NeuralType]]:
#         return {
#             'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
#             'a_sig_length': NeuralType(tuple('B'), LengthsType()),
#             'transcripts': NeuralType(('B', 'T'), LabelsType()),
#             'transcript_length': NeuralType(tuple('B'), LengthsType()),
#             'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
#             'target_lang_id': NeuralType(('B', 'T', 'D'), LabelsType())
#         }
    
#     def __init__(self, tokenizer, cfg):
#         super().__init__()
#         self.tokenizer = TokenizerWrapper(tokenizer)
#         self.load_audio = AudioSamples(fault_tolerant=True)
#         self.cfg = cfg
#         self.num_languages = cfg.get('num_languages', 128)
#         self.num_sample_per_mel_frame = cfg.get('num_sample_per_mel_frame', 160)
#         self.num_mel_frame_per_asr_frame = cfg.get('num_mel_frame_per_asr_frame', 8)
    
#         # Track unique languages seen during training
#         self.seen_languages = set()
#         self.language_to_index = {}
#         self.next_lang_idx = 0

#     def _get_language_index(self, language_code: str) -> int:
#         """
#         Maps language codes to indices dynamically.
#         """
#         if language_code not in self.language_to_index:
#             if self.next_lang_idx >= self.num_languages:
#                 raise ValueError(f"Number of unique languages ({self.next_lang_idx + 1}) exceeds maximum allowed ({self.num_languages})")
#             self.language_to_index[language_code] = self.next_lang_idx
#             self.next_lang_idx += 1
#             self.seen_languages.add(language_code)
#         return self.language_to_index[language_code]

#     def lang_to_target_lang(self, cut, num_languages: int, num_sample_per_mel_frame: int, num_mel_frame_per_asr_frame: int):
#         """
#         Create language target tensor for the sequence.
#         Similar to speaker_to_target but for languages.
#         """
#         # Calculate encoder output length
#         encoder_hidden_len = self.get_hidden_length_from_sample_length(
#             cut.num_samples,
#             num_sample_per_mel_frame,
#             num_mel_frame_per_asr_frame
#         )
        
#         # Initialize language target matrix
#         mask = np.zeros((num_languages, encoder_hidden_len))
        
#         # Get language index
#         lang_id = self._get_language_index(cut.supervisions[0].language)
#         # Set the corresponding language ID to 1 for all time steps
#         mask[lang_id, :] = 1
            
#         return mask

#     # this function is taken from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length        
#     def get_hidden_length_from_sample_length(self,
#         num_samples: int, 
#         num_sample_per_mel_frame: int = 160, 
#         num_mel_frame_per_asr_frame: int = 8
#     ) -> int:
#         """ 
#         Calculate the hidden length from the given number of samples.

#         This function computes the number of frames required for a given number of audio samples,
#         considering the number of samples per mel frame and the number of mel frames per ASR frame.

#         Parameters:
#             num_samples (int): The total number of audio samples.
#             num_sample_per_mel_frame (int, optional): The number of samples per mel frame. Default is 160.
#             num_mel_frame_per_asr_frame (int, optional): The number of mel frames per ASR frame. Default is 8.

#         Returns:
#             hidden_length (int): The calculated hidden length in terms of the number of frames.
#         """
#         mel_frame_count = math.ceil((num_samples + 1) / num_sample_per_mel_frame)
#         hidden_length = math.ceil(mel_frame_count / num_mel_frame_per_asr_frame)
#         return int(hidden_length)

    
#     def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
#         audio, audio_lens, cuts = self.load_audio(cuts)
#         tokens = [torch.as_tensor(self.tokenizer(c.supervisions[0].text, c.supervisions[0].language)) 
#                  for c in cuts]
        
#         # Create language targets
#         lang_targets = [torch.transpose(torch.as_tensor(self.lang_to_target_lang(
#                     c,
#                     self.num_languages,
#                     self.num_sample_per_mel_frame,
#                     self.num_mel_frame_per_asr_frame,
#                 ),
#                 dtype=torch.float32
#             ),
#             0,
#             1
#         ) for c in cuts]
        
#         # Create final tensors
#         token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
#         tokens = collate_vectors(tokens, padding_value=0)
#         lang_targets = collate_matrices(lang_targets)
        
#         return (
#             audio,           # Audio signal
#             audio_lens,      # Audio lengths
#             tokens,          # Text tokens
#             token_lens,      # Token lengths
#             lang_targets,    # Language targets
#         )

# class TokenizerWrapper:
#     """
#     Provide a unified interface for NeMo Tokenizer, AggregateTokenizer, and (char) Parser.
#     """
#     def __init__(self, tokenizer):
#         self._tokenizer = tokenizer
#         if isinstance(tokenizer, AggregateTokenizer):
#             self._impl = self._call_agg_tokenizer
#         elif isinstance(tokenizer, TokenizerSpec):
#             self._impl = self._call_tokenizer
#         else:
#             self._impl = self._call_parser
    
#     def __call__(self, text: str, lang: Optional[str] = None):
#         return self._impl(text, lang)
    
#     def _call_agg_tokenizer(self, text: str, lang: Optional[str] = None):
#         assert lang is not None, "Expected 'lang' to be set for AggregateTokenizer."
#         return self._tokenizer.text_to_ids(text, lang)
    
#     def _call_tokenizer(self, text: str, lang: Optional[str] = None):
#         return self._tokenizer.text_to_ids(text)
    
#     def _call_parser(self, text: str, lang: Optional[str] = None):
#         return self._tokenizer(text)