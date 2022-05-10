import os
import re
from lm_eval.base import rf, PerplexityTask
from lm_eval.utils import sh
import torch

from best_download import download_file


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation  # GEO: TODO: What if string ends with punctuation? (e.g. "bla .") Isn't replace(" .", ".") more general?
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


class WikiText(PerplexityTask):
    VERSION = 0

    def __init__(self, cache_dir=""):
        self.cache_dir = cache_dir
        super().__init__()

    def download(self):
        cache_dir = (
            self.cache_dir
            if self.cache_dir
            else os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data")
        )
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if not os.path.exists(cache_dir + "/wikitext/wikitext-2-raw/wiki.valid.raw"):
                os.makedirs(cache_dir + "/wikitext", exist_ok=True)
                download_file(
                    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
                    local_file=cache_dir + "/wikitext/wikitext-2-raw-v1.zip",
                    expected_checksum="ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11",
                )
                sh(f"cd {cache_dir}/wikitext && unzip wikitext-2-raw-v1.zip")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.cache_dir = cache_dir

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def has_validation_docs(self):
        return True

    def has_train_docs(self):
        return True

    def has_test_docs(self):
        return True

    def docs_for_split(self, split):
        ret = []
        for line in (
            open(self.cache_dir + f"/wikitext/wikitext-2-raw/wiki.{split}.raw").read().split("\n")
        ):
            rline = line.replace("= = =", "===").replace("= =", "==").strip()
            if rline.startswith("= ") and rline.strip().endswith(" ="):
                s = "\n".join(ret)
                if s.strip():
                    yield s
                ret = []
            ret.append(line)
        yield "\n".join(ret)

    def validation_docs(self):
        return self.docs_for_split("valid")

    def train_docs(self):
        return self.docs_for_split("train")

    def test_docs(self):
        return self.docs_for_split("test")

    def doc_to_target(self, doc):
        return wikitext_detokenizer(doc)

    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(re.split(r"\s+", doc))


class WikiText103(WikiText):
    def download(self):
        cache_dir = (
            self.cache_dir
            if self.cache_dir
            else os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data")
        )
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if not os.path.exists(cache_dir + "/wikitext/wikitext-103-raw/wiki.valid.raw"):
                os.makedirs(cache_dir + "/wikitext", exist_ok=True)
                download_file(
                    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
                    local_file=cache_dir + "/wikitext/wikitext-103-raw-v1.zip",
                )
                sh(f"cd {cache_dir}/wikitext && unzip wikitext-103-raw-v1.zip")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.cache_dir = cache_dir

    def docs_for_split(self, split):
        ret = []
        for line in (
            open(self.cache_dir + f"/wikitext/wikitext-103-raw/wiki.{split}.raw").read().split("\n")
        ):
            rline = line.replace("= = =", "===").replace("= =", "==").strip()
            if rline.startswith("= ") and rline.strip().endswith(" ="):
                s = "\n".join(ret)
                if s.strip():
                    yield s
                ret = []
            ret.append(line)
        yield "\n".join(ret)
