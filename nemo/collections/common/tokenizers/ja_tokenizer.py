import re
from typing import List

from sacremoses import MosesDetokenizer


class JADetokenizer:
    def __init__(self):
        self.moses_detokenizer = MosesDetokenizer(lang="ja")

    def detokenize(self, translation: List[str]):
        translation = self.moses_detokenizer.detokenize(translation)
        return re.sub('▁', ' ', re.sub(' ', '', translation))
