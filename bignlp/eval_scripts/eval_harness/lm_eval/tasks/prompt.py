import json
import re
import os

from best_download import download_file
from copy import deepcopy

from tqdm.auto import tqdm
from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.core import Dataset

from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity
from lm_eval.utils import sh

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s | %(name)-7s | %(levelname)-8s: %(message)s"
)
logger = logging.getLogger(__name__)


# Modified based on
# https://github.com/NVIDIA/NeMo/blob/e165f653d47c4faf89ecd97720803b8ef964a6ce/nemo/collections/nlp/data/language_modeling/megatron/gpt_prompt_learning_dataset.py#L28
class GPTPromptLearningDataset(Dataset):
    """
    The dataset class for prompt-tuning or p-tuning pretrained GPT models.
    """

    def __init__(
        self,
        datasets,
        tokenizer,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: str,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = False,
        for_train: bool = False,
    ):
        self.tokenizer = tokenizer
        self.virtual_prompt_source = virtual_prompt_source
        self.task_templates = task_templates
        self.pseudo_tokens = pseudo_tokens
        self.pseudo_token_ids = set(self.tokenizer.tokens_to_ids(self.pseudo_tokens))
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.for_train = for_train
        self.examples = []

        assert self.min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert self.max_seq_length > 0, "Max sequence length should be greater than 0"

        logger.info("Loading and tokenizing dataset ... ")

        # Datasets is just a list of json dicts
        if isinstance(datasets[0], dict):
            self.load_data(datasets)

        # Datasets are a list of file path strings to .json or .jsonl files
        elif isinstance(datasets[0], str):
            for path in datasets:
                dataset = open(path, 'r', encoding='utf-8')
                self.load_data(dataset)
        else:
            raise ValueError("Datasets must be a list of dicts or a list of filepath strings")

    def load_data(self, dataset):
        """
        Loads a dataset by filling in the task templates specified in the config file
        with the information from each training/inference example. Converts all input
        text into token ids. Also replaces the <|VIRTUAL_PROMPT_#|> placeholders in
        the task templates with the actual virtual prompt token ids.

        params:
            dataset: A list of json objects or a dictionary objects each
                     containing the information needed for a training example
        """
        skipped = 0

        for json_line in tqdm(dataset):

            # Read example dict or load the information for a single example from .json file
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)

            taskname = doc["taskname"]
            prompt_template = self.task_templates[taskname]["prompt_template"]
            prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
            total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
            virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
            truncation_field = self.task_templates[taskname]['truncate_field']
            answer_only_loss = self.task_templates[taskname]["answer_only_loss"]
            answer_field = self.task_templates[taskname]["answer_field"]

            input_example = prompt_template

            self._input_sanity_checks(
                total_virtual_tokens,
                virtual_token_splits,
                prompt_template,
                prompt_template_fields,
                truncation_field,
                answer_only_loss,
                answer_field,
                doc,
            )

            # Format the input example according to the template
            input_example = self._insert_text_in_template(input_example, prompt_template_fields, doc)
            input_example = self._insert_virtual_token_placeholders(input_example, virtual_token_splits)
            input_ids = self.tokenizer.text_to_ids(input_example)

            # Add BOS/EOS if desired, adds EOS by default
            if self.add_bos:
                input_ids = [self.tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [self.tokenizer.eos_id]

            # Try to truncate input text to fit into the max sequence length
            if len(input_ids) > self.max_seq_length:
                input_ids = self._truncate_input(truncation_field, input_ids, taskname, doc)

            # Skip example if the final length doesn't fit length requirements even after truncation
            if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
                if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                    taskname_id = self.tokenizer.text_to_ids(taskname)

                elif self.virtual_prompt_source == VirtualPromptSource.PROMPT_TABLE:
                    taskname_id = self.task_templates[taskname]["task_id_num"]

                answer_start_idx = self._find_answer_start(taskname, input_ids, answer_field, doc)

                self.examples.append((taskname_id, input_ids, answer_start_idx))
            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')

    def _input_sanity_checks(
        self,
        total_virtual_tokens,
        virtual_token_splits,
        prompt_template,
        prompt_template_fields,
        truncation_field,
        answer_only_loss,
        answer_field,
        doc,
    ):
        # Sanity check amount of virtual token
        assert total_virtual_tokens > 0, "There should be at least one virtual prompt token"
        assert (
            total_virtual_tokens < self.max_seq_length
        ), "virtual prompt tokens should not exceed max sequence length"

        # Make sure virtual token splits add up to the total number of virtual tokens
        assert (
            sum(virtual_token_splits) == total_virtual_tokens
        ), "Sum of prompt token split values must equal total number of prompt tokens"

        # Make sure number of virtual prompt locations match the number of virtual prompt splits
        assert prompt_template.count('<|VIRTUAL_PROMPT_') == len(
            virtual_token_splits
        ), "The number of '<|VIRTUAL_PROMPT_n|>' markers and the number of prompt token splits must match"

        # Check if input example has fields not present in template
        keys_not_in_template = list(set(doc.keys()) - set(prompt_template_fields) - set(['taskname']))
        assert (
            len(keys_not_in_template) == 0
        ), f"Examples in your dataset contain the fields: {keys_not_in_template} that are not in the task template."

        # Check that answer field checks if answer_only_loss was set to True
        if answer_only_loss and self.for_train:
            assert answer_field is not None, "If answer_only_loss=True, an answer_field must be given"
            assert (
                answer_field in doc.keys()
            ), f"answer_only_loss=True but the given answer_field '{answer_field}' is not in data json"
            assert truncation_field != answer_field, "Answer field and truncation field should not match"

            answer_placeholder = "{" + answer_field + "}"
            answer_placeholder_len = len(answer_placeholder)
            placeholder_start = len(prompt_template) - answer_placeholder_len
            assert prompt_template[placeholder_start:] == answer_placeholder, "Answer field must be at prompt end"

    def _insert_text_in_template(self, input_example, prompt_template_fields, doc):
        """ Format the input example according to the template """
        for field in prompt_template_fields:
            if field in doc.keys():
                field_text = doc[field]
                input_example = input_example.replace('{' + field + '}', field_text)

            # If some fields from the template aren't present, e.g. {answer} during inference
            # just remove that field from the template, leaving the space blank
            else:
                input_example = input_example.replace('{' + field + '}', "")

        return input_example

    def _insert_virtual_token_placeholders(self, input_example, virtual_token_splits):
        """ Insert the correct number of pseudo tokens at the <|virtual_PROMPT_n|> markers """
        total_inserted_tokens = 0

        for idx in range(len(virtual_token_splits)):
            split_start = total_inserted_tokens
            split_end = total_inserted_tokens + virtual_token_splits[idx]
            pseudo_tokens_for_split = "".join(self.pseudo_tokens[split_start:split_end])
            input_example = input_example.replace(f'<|VIRTUAL_PROMPT_{idx}|>', pseudo_tokens_for_split)
            total_inserted_tokens = split_end

        return input_example

    def _truncate_input(self, truncation_field, input_ids, taskname, doc):
        """ Try to truncate input text to fit into the max sequence length """
        logging.info(
            f"Input greater than max sequence length. Attempting to truncate: '{truncation_field}' in task: '{taskname}'"
        )

        # Truncate the text ids in this part of input to try and fit max sequence length
        if truncation_field is not None and truncation_field in doc.keys():
            truncation_length = len(input_ids) - self.max_seq_length
            field_text = doc[truncation_field]
            field_text = self._add_leading_space(taskname, truncation_field, field_text)

            # Truncate field text
            field_text_ids = self.tokenizer.text_to_ids(field_text)
            truncated_text_ids = field_text_ids[: -min(truncation_length, len(field_text_ids))]

            # Replace original text ids with truncated text ids
            field_start, field_end = find_subsequence_location(input_ids, field_text_ids)
            input_ids = input_ids[:field_start] + truncated_text_ids + input_ids[field_end + 1 :]

        return input_ids

    def _find_answer_start(self, taskname, input_ids, answer_field, doc):
        """ Find the token ids corresponding to the answer start, for loss masking purposes.
            Assumes the answer is always at the end of the prompt.
        """
        answer_text = doc[answer_field]
        answer_text = self._add_leading_space(taskname, answer_field, answer_text)
        answer_text_ids = self.tokenizer.text_to_ids(answer_text)
        num_answer_text_ids = len(answer_text_ids)

        if self.add_eos:
            num_answer_text_ids += 1

        answer_start_idx = len(input_ids) - num_answer_text_ids

        return answer_start_idx

    def _add_leading_space(self, taskname, field_name, field_text):
        """ Add leading space to text if there is a space before it in the template """
        prompt_template = self.task_templates[taskname]["prompt_template"]
        field_text_start = prompt_template.find("{" + field_name + "}")
        if field_text_start != 0 and prompt_template[field_text_start - 1] == " ":
            field_text = " " + field_text

        return field_text

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def get_all_examples(self, tokens_to_generate):
        """
        Used for loading inference data.
        """
        task_id_nums, input_ids, answer_starts = zip(*self.examples)
        input_lengths = torch.cuda.LongTensor([len(inputs) for inputs in input_ids])
        task_id_nums = torch.cuda.LongTensor(task_id_nums)
        batch_max = input_lengths.max().item()
        batch_max += tokens_to_generate

        input_ids, _ = self.pad_batch_and_build_loss_mask(input_ids, batch_max, answer_starts)
        input_ids = input_ids.cuda()
        input_ids = torch.cuda.LongTensor(input_ids)

        return task_id_nums, (input_ids, input_lengths)



class Prompt(Task):
    VERSION = 0

    def __init__(self, model, dataset_paths, disable_special_tokens=False):
        super().__init__()
        self.tokenizer = model.tokenizer
        self.disable_special_tokens = disable_special_tokens
        self.prompt_dataset = GPTPromptLearningDataset(
            datasets=dataset_paths,
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

    def fewshot_context(
        self, doc, num_fewshot, provide_description, rnd, filter_shot_examples=False, **kwargs
    ):
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
        description = (
            (raw_description + "\n===\n\n") if provide_description and raw_description else ""
        )

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
