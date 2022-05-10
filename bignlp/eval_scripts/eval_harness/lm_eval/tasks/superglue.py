"""
To-do:
    - WSC requires free-form generation
    - ReCoRD
"""
import numpy as np
import sklearn
import transformers.data.metrics.squad_metrics as squad_metrics
from .common import HFTask, yesno
from lm_eval.base import rf
from ..metrics import mean, acc_all, metric_max_over_ground_truths
from ..utils import general_detokenize


class BoolQ(HFTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "boolq"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out actual description
        return "Read the following passages and answer each question with a yes or a no."

    def doc_to_text(self, doc):
        return f"{doc['passage']}\nQuestion: {doc['question']}\nAnswer:"

    def doc_to_target(self, doc):
        return " " + yesno(doc["label"])

    def construct_requests(self, doc, ctx):

        ll_yes, *_ = rf.loglikelihood(ctx, " yes")
        ll_no, *_ = rf.loglikelihood(ctx, " no")

        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]

        acc = 1.0 if (ll_yes > ll_no) == gold else 0.0

        return {"acc": acc}

    def serialize_results(self, doc, results):
        ll_yes, ll_no = results
        return {
            "format": self.doc_to_text(doc) + " {yes/no}",
            "model_choice": "yes" if ll_yes > ll_no else "no",
            "gold_choice": "yes" if doc["label"] else "no",
            "likelihoods": {"yes": ll_yes, "no": ll_no},
        }

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class CommitmentBank(HFTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "cb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out actual description
        return (
            "Given a premise and a hypothesis, classify whether the author of the premise is committed"
            "to the truth of the hypothesis. The three possible labels are true, false or neither."
        )

    def doc_to_text(self, doc):
        return "{}\nQuestion: {}. True, False or Neither?\nAnswer:".format(
            doc["premise"],
            doc["hypothesis"],
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "True", 1: "Neither", 2: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _, _, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _, _, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _, _, _ = rf.loglikelihood(ctx, " False")

        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        acc = 1.0 if pred == gold else 0.0

        return {"acc": acc, "f1": (pred, gold)}

    def serialize_results(self, doc, results):
        return {
            "gold_choice": doc["label"],
            "model_output": results,
            "question": doc["premise"]
            + "\nQuestion: "
            + doc["hypothesis"]
            + ". True, False or Neither?\nAnswer:",
        }

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    @classmethod
    def cb_multi_fi(cls, items):
        preds, golds = zip(*items)
        preds = np.array(preds)
        golds = np.array(golds)
        f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
        f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
        f13 = sklearn.metrics.f1_score(y_true=golds == 2, y_pred=preds == 2)
        avg_f1 = mean([f11, f12, f13])
        return avg_f1

    def aggregation(self):
        return {
            "acc": mean,
            "f1": self.cb_multi_fi,
        }


class Copa(HFTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "copa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out actual description
        return (
            "Given a premise and one alternative with a causal relation to the premise and another without,"
            "choose the more plausible alternative"
        )

    def doc_to_text(self, doc):
        # Drop the period
        connector = {
            "cause": "because",
            "effect": "therefore",
        }[doc["question"]]
        return doc["premise"].strip()[:-1] + f" {connector}"

    def doc_to_target(self, doc):
        correct_choice = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
        # Connect the sentences
        return " " + self.convert_choice(correct_choice)

    def construct_requests(self, doc, ctx):
        choice1 = " " + self.convert_choice(doc["choice1"])
        choice2 = " " + self.convert_choice(doc["choice2"])

        ll_choice1, _, _, _ = rf.loglikelihood(ctx, choice1)
        ll_choice2, _, _, _ = rf.loglikelihood(ctx, choice2)

        return ll_choice1, ll_choice2

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        acc = 1.0 if pred == gold else 0.0

        return {"acc": acc}

    def serialize_results(self, doc, results):
        return {
            "gold_choice": doc["label"],
            "model_output": results,
            "premise": doc["premise"],
        }

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]


class MultiRC(HFTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "multirc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out actual description
        return "READING COMPREHENSION ANSWER KEY"

    def doc_to_text(self, doc):
        return f"{doc['paragraph']}\nQuestion: {doc['question']}\nAnswer:"

    def doc_to_target(self, doc):
        return " " + self.format_answer(answer=doc["answer"], label=doc["label"])

    @staticmethod
    def format_answer(answer, label):
        label_str = "yes" if label else "no"
        return f"{label_str}, {answer}"

    def construct_requests(self, doc, ctx):
        true_choice = self.format_answer(answer=doc["answer"], label=True)
        false_choice = self.format_answer(answer=doc["answer"], label=False)

        ll_true_choice, _, _, _ = rf.loglikelihood(ctx, f" {true_choice}")
        ll_false_choice, _, _, _ = rf.loglikelihood(ctx, f" {false_choice}")

        return ll_true_choice, ll_false_choice

    def process_results(self, doc, results):
        pred = np.argmax(results)
        return {"acc": (pred, doc)}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": acc_all}


class ReCoRD(HFTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "record"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out actual description
        return ""

    def training_docs(self):
        # In ReCoRD, each doc manifests multiple "examples" in the context of few shot example packing.
        # Each doc consists of multiple answer candidates, each of which is scored yes/no.
        if self._training_docs is None:
            self._training_docs = []
            for doc in self.data["train"]:
                self._training_docs.append(self._process_doc(doc))
        return self._training_docs

    def validation_docs(self):
        # See: training_docs
        for doc in self.data["validation"]:
            yield self._process_doc(doc)

    @classmethod
    def _process_doc(cls, doc):
        return {
            "passage": doc["passage"],
            "query": doc["query"],
            "entities": sorted(list(set(doc["entities"]))),
            "answers": sorted(list(set(doc["answers"]))),
        }

    def doc_to_text(self, doc):
        initial_text, *highlights = doc["passage"].strip().split("\n@highlight\n")
        text = initial_text + "\n\n"
        for highlight in highlights:
            text += f"  - {highlight}.\n"
        return text

    @classmethod
    def format_answer(cls, query, entity):
        return f"  - {query}".replace("@placeholder", entity)

    def doc_to_target(self, doc):
        # We only output the first correct entity in a doc
        return self.format_answer(query=doc["query"], entity=doc["answers"][0])

    def construct_requests(self, doc, ctx):
        requests = [
            rf.loglikelihood(ctx, self.format_answer(query=doc["query"], entity=entity))
            for entity in doc["entities"]
        ]
        return requests

    def process_results(self, doc, results):
        # ReCoRD's evaluation is actually deceptively simple:
        # - Pick the maximum likelihood prediction entity
        # - Evaluate the accuracy and token F1 PER EXAMPLE
        # - Average over all examples
        max_idx = np.argmax(np.array([result[0] for result in results]))

        prediction = doc["entities"][max_idx]
        gold_label_set = doc["answers"]
        f1 = metric_max_over_ground_truths(squad_metrics.compute_f1, prediction, gold_label_set)
        em = metric_max_over_ground_truths(squad_metrics.compute_exact, prediction, gold_label_set)

        return {
            "f1": f1,
            "em": em,
        }

    def higher_is_better(self):
        return {
            "f1": True,
            "em": True,
        }

    def aggregation(self):
        return {
            "f1": mean,
            "em": mean,
        }


class WordsInContext(HFTask):
    VERSION = 0
    DATASET_PATH = "super_glue"
    DATASET_NAME = "wic"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out actual description
        return ""

    def doc_to_text(self, doc):
        return (
            "Sentence 1: {}\nSentence 2: {}\nQuestion: Is the word '{}' used in the same way in the"
            " two sentences above?\nAnswer:".format(
                doc["sentence1"],
                doc["sentence2"],
                doc["sentence1"][doc["start1"] : doc["end1"]],
            )
        )

    def doc_to_target(self, doc):
        return " {}".format({0: "no", 1: "yes"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_yes, _, _, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _, _, _ = rf.loglikelihood(ctx, " no")

        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]

        acc = 1.0 if (ll_yes > ll_no) == gold else 0.0

        return {"acc": acc}

    def serialize_results(self, doc, results):
        return {
            "gold_choice": doc["label"],
            "model_output": results,
            "question": "Sentence 1: "
            + doc["sentence1"]
            + "\nSentence 2: "
            + doc["sentence2"]
            + "\nQuestion: Is the word "
            + doc["sentence1"][doc["start1"] : doc["end1"]]
            + " used in the same way in the two sentences above?\nAnswer:",
        }

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class SGWinogradSchemaChallenge(HFTask):
    VERSION = 0
    # Note: This implementation differs from Fig G.32 because this is the SuperGLUE,
    #       binary version of the task.
    DATASET_PATH = "super_glue"
    DATASET_NAME = "wsc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                # GPT-3 Paper's format only uses positive examples for fewshot "training"
                self._training_docs = [doc for doc in self.data["train"] if doc["label"]]
            return self._training_docs

    def fewshot_description(self):
        return (
            "Final Exam with Answer Key\n"
            "Instructions: Please carefully read the following passages. "
            "For each passage, you must identify which noun the pronoun marked in *bold*"
            " refers to.\n====="
        )

    def doc_to_text(self, doc):
        raw_passage = doc["text"]
        # NOTE: HuggingFace span indices are word-based not character-based.
        pre = " ".join(raw_passage.split()[: doc["span2_index"]])
        post = raw_passage[len(pre) + len(doc["span2_text"]) + 1 :]
        passage = general_detokenize(pre + " *{}*".format(doc["span2_text"]) + post)
        noun = doc["span1_text"]
        pronoun = doc["span2_text"]
        text = (
            f"Passage: {passage}\n"
            + f'Question: In the passage above, does the pronoun "*{pronoun}*" refer to "*{noun}*"?\n'
            + "Answer:"
        )
        return text

    def doc_to_target(self, doc):
        return " " + yesno(doc["label"])

    def construct_requests(self, doc, ctx):

        ll_yes, _, _, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _, _, _ = rf.loglikelihood(ctx, " no")

        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]

        acc = 1.0 if (ll_yes > ll_no) == gold else 0.0

        return {"acc": acc}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
