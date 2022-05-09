import json
import re
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity
from lm_eval.utils import sh
from best_download import download_file
import os


class LAMBADA(Task):
    VERSION = 0

    def __init__(self, cache_dir=""):
        self.cache_dir = cache_dir
        super().__init__()

    def download(self):
        path = (
            self.cache_dir
            if self.cache_dir
            else os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data")
        )
        path = os.path.join(path, "lambada")
        sh("mkdir -p " + path)

        try:
            if not os.path.exists(path + "/lambada_test.jsonl"):
                download_file(
                    "http://eaidata.bmk.sh/data/lambada_test.jsonl",
                    local_file=path + "/lambada_test.jsonl",
                    expected_checksum="4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226",
                )
        except:
            # fallback - for some reason best_download doesnt work all the time here
            sh(
                "wget http://eaidata.bmk.sh/data/lambada_test.jsonl -O data/lambada/lambada_test.jsonl"
            )
            sh(
                'echo "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226  data/lambada/lambada_test.jsonl" | sha256sum --check'
            )

        self.cache_dir = path

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        pass

    def validation_docs(self):
        path = self.cache_dir

        # with open("data/lambada/lambada_test.jsonl") as fh:
        with open(path + "/lambada_test.jsonl") as fh:
            for line in fh:
                yield json.loads(line)

    def test_docs(self):
        pass

    def preprocess(self, text):
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("’", "'")
        text = text.replace("‘", "'")
        return text

    def doc_to_text(self, doc):
        return "\n" + self.preprocess(doc["text"].rsplit(" ", 1)[0]).strip()

    def doc_to_target(self, doc):
        return " " + self.preprocess(doc["text"].rsplit(" ", 1)[1])

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def construct_requests(self, doc, ctx):
        ll, is_greedy, greedy_toks, cont_toks = rf.loglikelihood(ctx, self.doc_to_target(doc))

        return ll, is_greedy, greedy_toks, cont_toks

    def process_results(self, doc, results):
        ll, is_greedy, *_ = results

        return {"ppl": ll, "acc": int(is_greedy)}

    def serialize_results(self, doc, results):
        *_, greedy_toks, cont_toks = results
        return {
            "prompt": self.doc_to_text(doc),
            "gold_answer": [x.replace("Ġ", " ") for x in cont_toks],
            "model_answer": [x.replace("Ġ", " ") for x in greedy_toks],
        }

    def aggregation(self):
        return {"ppl": perplexity, "acc": mean}

    def higher_is_better(self):
        return {"ppl": False, "acc": True}
