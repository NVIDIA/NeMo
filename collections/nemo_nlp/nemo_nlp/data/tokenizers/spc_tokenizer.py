import sentencepiece as spm
from .tokenizer_spec import TokenizerSpec


class SentencePieceTokenizer(TokenizerSpec):
    def __init__(self, model_path):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model_path)
        self.original_vocab_size = self.tokenizer.get_piece_size()
        self.vocab_size = self.tokenizer.get_piece_size()
        self.special_tokens = {}
        self.special_token_ids = {}

    def text_to_tokens(self, text):
        tokens = []
        idx = 0
        last_idx = 0

        while 1:
            indices = {}

            for token in self.special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue

            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            tokens.extend(self.tokenizer.encode_as_pieces(text[idx:next_idx]))
            tokens.append(next_token)
            idx = next_idx + len(next_token)

        tokens.extend(self.tokenizer.encode_as_pieces(text[idx:]))
        return tokens

    def tokens_to_text(self, tokens):
        return self.tokenizer.decode_pieces(tokens)

    def text_to_ids(self, text):
        ids = []
        idx = 0
        last_idx = 0

        while 1:
            indices = {}

            for token in self.special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue

            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self.special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
        return ids

    def ids_to_text(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self.special_token_ids:
                text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self.special_token_ids[id] + " "
                last_i = i + 1

        text += self.tokenizer.decode_ids(ids[last_i:])
        return text.strip()

    def token_to_id(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]

        return self.tokenizer.piece_to_id(token)

    def tokens_to_ids(self, tokens):
        ids = []

        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.tokenizer.piece_to_id(token))

        return ids

    def ids_to_tokens(self, ids):
        tokens = []

        for id in ids:
            if id >= self.original_vocab_size:
                tokens.append(self.special_token_ids[id])
            else:
                tokens.append(self.tokenizer.id_to_piece(id))

        return tokens

    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            if self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id():
                self.special_tokens[token] = self.vocab_size
                self.special_token_ids[self.vocab_size] = token
                self.vocab_size += 1
