import numpy as np
from lm_eval.base import MultipleChoiceTask, rf
from ..metrics import mean
from .common import HFTask


class PiQA(HFTask, MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "piqa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def _convert_standard(self, doc):
        out_doc = {
            "goal": doc["goal"],
            "choices": [doc["sol1"], doc["sol2"]],
            "gold": doc["label"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["goal"]
