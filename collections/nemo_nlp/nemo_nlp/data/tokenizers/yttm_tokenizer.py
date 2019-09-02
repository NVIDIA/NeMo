import youtokentome as yttm
from .tokenizer_spec import TokenizerSpec


class YouTokenToMeTokenizer(TokenizerSpec):
    def __init__(self, model_path):
        self.tokenizer = yttm.BPE(model=model_path)
        self.vocab_size = len(self.tokenizer.vocab())
        self.special_tokens = self.tokens_to_ids(
            ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])

    def text_to_tokens(self, text):
        return self.tokenizer.encode(text, output_type=yttm.OutputType.SUBWORD)

    def tokens_to_text(self, tokens):
        return self.ids_to_text(self.tokens_to_ids(tokens))

    def text_to_ids(self, text):
        return self.tokenizer.encode(text, output_type=yttm.OutputType.ID)

    def ids_to_text(self, ids):
        ids_ = [id_ for id_ in ids if id_ not in self.special_tokens]
        return self.tokenizer.decode([ids_])[0]

    def tokens_to_ids(self, tokens):
        return [self.tokenizer.subword_to_id(token) for token in tokens]

    def ids_to_tokens(self, ids):
        ids_ = [id_ for id_ in ids if id_ not in self.special_tokens]
        return [self.tokenizer.id_to_subword(id_) for id_ in ids_]

    def pad_id(self):
        return self.tokenizer.subword_to_id("<PAD>")

    def bos_id(self):
        return self.tokenizer.subword_to_id("<BOS>")

    def eos_id(self):
        return self.tokenizer.subword_to_id("<EOS>")
