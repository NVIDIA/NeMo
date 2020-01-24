from .tokenizer_spec import TokenizerSpec


class WordTokenizer(TokenizerSpec):
    def __init__(self, vocab_path):

        vocab_list = open(vocab_path, "r").readlines()
        self.vocab = {vocab_list[i].strip(): i for i in range(len(vocab_list))}
        for special_token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            if special_token not in self.vocab:
                self.vocab[special_token] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.special_tokens = self.tokens_to_ids(
            ["<PAD>", "<UNK>", "<BOS>", "<EOS>"])

    def text_to_tokens(self, text):
        token_candidates = text.strip().split()
        tokens = []
        for token in token_candidates:
            if token in self.vocab:
                tokens.append(token)
            else:
                tokens.append("<UNK>")
        return tokens

    def tokens_to_text(self, tokens):
        return self.ids_to_text(self.tokens_to_ids(tokens))

    def text_to_ids(self, text):
        return [self.vocab[token] for token in self.text_to_tokens(text)]

    def ids_to_text(self, ids):
        ids_ = [id_ for id_ in ids if id_ not in self.special_tokens]
        return " ".join(self.ids_to_tokens(ids_))

    def tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def ids_to_tokens(self, ids):
        return [self.inv_vocab[id] for id in ids]

    def pad_id(self):
        return self.vocab["<PAD>"]

    def bos_id(self):
        return self.vocab["<BOS>"]

    def eos_id(self):
        return self.vocab["<EOS>"]
