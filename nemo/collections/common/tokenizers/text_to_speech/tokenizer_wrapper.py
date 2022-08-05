from nemo.collections.common.tokenizers import TokenizerSpec

from nemo.collections.common.tokenizers.text_to_speech.g2ps import EnglishG2p
from nemo.collections.common.tokenizers.text_to_speech.tokenizers import EnglishPhonemesTokenizer

__all__ = ['TextToSpeechTokenizer']

class TextToSpeechTokenizer(TokenizerSpec):
    def __init__(self, phoneme_dict, heteronyms):
        self.g2p = EnglishG2p(phoneme_dict=phoneme_dict, heteronyms=heteronyms)
        self.tokenizer = EnglishPhonemesTokenizer(
            self.g2p, stresses=True, chars=True, pad_with_space=True, add_blank_at=True)
        self.vocab_size = len(self.tokenizer.tokens)

    def text_to_ids(self, text):
        return self.tokenizer.encode(text)
    
    def text_to_tokens(self, text):
        return self.g2p(text)

    def tokens_to_text(self, tokens):
        pass

    def tokens_to_ids(self, tokens):
        pass

    def ids_to_tokens(self, ids):
        pass

    def ids_to_text(self, ids):
        pass

    @property
    def pad_id(self):
        return self.tokenizer.pad

    @property
    def bos_id(self):
        return self.tokenizer.pad

    @property
    def eos_id(self):
        return self.tokenizer.pad
