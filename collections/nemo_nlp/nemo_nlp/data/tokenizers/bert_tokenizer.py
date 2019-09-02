from .tokenizer_spec import TokenizerSpec
from pytorch_transformers import BertTokenizer
import re


def handle_quotes(text):
    text_ = ""
    quote = 0
    i = 0
    while i < len(text):
        if text[i] == "\"":
            if quote % 2:
                text_ = text_[:-1] + "\""
            else:
                text_ += "\""
                i += 1
            quote += 1
        else:
            text_ += text[i]
        i += 1
    return text_


def remove_spaces(text):
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace("[ ", "[")
    text = text.replace(" ]", "]")
    text = text.replace(" / ", "/")
    text = text.replace("„ ", "„")
    text = text.replace(" - ", "-")
    text = text.replace(" ' ", "'")
    text = re.sub(r'([0-9])( )([\.,])', '\\1\\3', text)
    text = re.sub(r'([\.,])( )([0-9])', '\\1\\3', text)
    text = re.sub(r'([0-9])(:)( )([0-9])', '\\1\\2\\4', text)
    text = text.replace(" %", "%")
    text = text.replace("$ ", "$")
    text = text.replace("\xa0", " ")
    text = re.sub(r'([^0-9])(,)([0-9])', '\\1\\2 \\3', text)
    return text


class NemoBertTokenizer(TokenizerSpec):
    def __init__(self, pretrained_model=None,
                 vocab_file=None,
                 do_lower_case=True,
                 max_len=None,
                 do_basic_tokenize=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if pretrained_model:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            if "uncased" not in pretrained_model:
                self.tokenizer.basic_tokenizer.do_lower_case = False
        else:
            self.tokenizer = BertTokenizer(vocab_file,
                                           do_lower_case,
                                           max_len,
                                           do_basic_tokenize,
                                           never_split)
        self.vocab_size = len(self.tokenizer.vocab)
        self.never_split = never_split

    def text_to_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return remove_spaces(handle_quotes(text.strip()))

    def token_to_id(self, token):
        return self.tokens_to_ids([token])[0]

    def tokens_to_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        tokens_clean = [t for t in tokens if t not in self.never_split]
        text = self.tokens_to_text(tokens_clean)
        return text

    def pad_id(self):
        return self.tokens_to_ids(["[PAD]"])[0]

    def bos_id(self):
        return self.tokens_to_ids(["[CLS]"])[0]

    def eos_id(self):
        return self.tokens_to_ids(["[SEP]"])[0]
