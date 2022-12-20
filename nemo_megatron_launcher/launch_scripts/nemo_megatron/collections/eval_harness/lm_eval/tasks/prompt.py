# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset


class Prompt(Task):
    VERSION = 0

    def __init__(self, model, dataset_paths, disable_special_tokens=False):
        super().__init__()
        self.tokenizer = model.tokenizer
        self.disable_special_tokens = disable_special_tokens
        self.prompt_dataset = GPTPromptLearningDataset(
            data=dataset_paths,
            tokenizer=model.tokenizer,
            virtual_prompt_source=model.virtual_prompt_source,
            task_templates=model.task_templates,
            pseudo_tokens=model.pseudo_tokens,
            pad_token_id=model.pad_token_id,
            max_seq_length=2048,
            min_seq_length=1,
            add_bos=False,
            add_eos=True,
            for_train=True,
        )

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        pass

    def validation_docs(self):
        for example in self.prompt_dataset:
            task_id, input_ids, answer_start_idx = example
            if self.disable_special_tokens:
                context = self.tokenizer.ids_to_text(input_ids[:answer_start_idx])
            else:
                context = self.tokenizer.tokens_to_text(self.tokenizer.ids_to_tokens(input_ids[:answer_start_idx]))
            doc = {
                "task_id": task_id,
                "context": context,
                "target": self.tokenizer.ids_to_text(input_ids[answer_start_idx:]),
            }
            yield doc

    def test_docs(self):
        pass

    def fewshot_context(self, doc, num_fewshot, provide_description, rnd, filter_shot_examples=False, **kwargs):
        """Construct and format full prompt string for a given sample, optionally including description and shot examples
        :param doc: document object corresponding to the sample under examination
        :param num_fewshot: number of examples to be included in the prompt
        :param provide_description: (bool), whether to prepend natural language description
        :param rnd: initialized random number generator object, e.g. rnd = random.Random(1337)
        :param filter_shot_examples: If True, will make sure to exclude certain samples from the prompt context, based
            on member `filter_shots` function
        :return: (shot_ids, context_str): tuple of (iterable of shot example IDs, string correspoding to context/prompt)
        """

        raw_description = self.fewshot_description()
        description = (raw_description + "\n===\n\n") if provide_description and raw_description else ""

        if num_fewshot == 0:
            labeled_examples = ""
            shot_ids = []
        else:
            raise NotImplementedError("No support for fewshots in prompt model evaluation.")

        example = self.doc_to_text(doc)  # the document of interest, main part of the prompt
        prompt_str = description + labeled_examples + example  # the formatted prompt string
        return shot_ids, prompt_str

    def doc_to_text(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        return doc["target"]

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def construct_requests(self, doc, ctx):
        ll, is_greedy, greedy_toks, cont_toks = rf.loglikelihood(ctx, self.doc_to_target(doc), doc["task_id"])

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
