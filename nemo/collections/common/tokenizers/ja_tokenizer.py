from sacremoses import MosesDetokenizer
import re
from typing import List

class JADetokenizer:
    def __init__(self):
        self.moses_detokenizer = MosesDetokenizer(lang="ja")

    def detokenize(self, translation: List[str]):
        translation re.sub('▁', ' ', re.sub(' ', '', "".join(translation)))
        return self.moses_detokenizer.detokenize(translation.split(" "))
