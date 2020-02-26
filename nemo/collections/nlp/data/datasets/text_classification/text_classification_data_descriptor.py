from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils import (
    calc_class_weights,
    get_intent_labels,
    get_label_stats,
    if_exist,
    process_imdb,
    process_jarvis_datasets,
    process_nlu,
    process_sst_2,
    process_thucnews,
)

__all__ = ['TextClassificationDataDesc']


class TextClassificationDataDesc:
    def __init__(self, dataset_name, data_dir, do_lower_case, eval_file_prefix=None):

        if dataset_name == 'sst-2':
            self.data_dir = process_sst_2(data_dir)
            self.num_labels = 2
            self.eval_file = self.data_dir + '/dev.tsv'

        elif dataset_name == 'imdb':
            self.num_labels = 2
            self.data_dir = process_imdb(data_dir, do_lower_case)
            self.eval_file = self.data_dir + '/test.tsv'
        elif dataset_name == 'thucnews':
            self.num_labels = 14
            self.data_dir = process_thucnews(data_dir)
            self.eval_file = self.data_dir + '/test.tsv'
        elif dataset_name.startswith('nlu-'):
            if dataset_name.endswith('chat'):
                self.data_dir = f'{data_dir}/ChatbotCorpus.json'
                self.num_labels = 2
            elif dataset_name.endswith('ubuntu'):
                self.data_dir = f'{data_dir}/AskUbuntuCorpus.json'
                self.num_labels = 5
            elif dataset_name.endswith('web'):
                data_dir = f'{data_dir}/WebApplicationsCorpus.json'
                self.num_labels = 8
            self.data_dir = process_nlu(data_dir, do_lower_case, dataset_name=dataset_name)
            self.eval_file = self.data_dir + '/test.tsv'
        elif dataset_name.startswith('jarvis'):
            self.data_dir = process_jarvis_datasets(
                data_dir, do_lower_case, dataset_name, modes=['train', 'test', 'eval'], ignore_prev_intent=False
            )

            intents = get_intent_labels(f'{self.data_dir}/dict.intents.csv')
            self.num_labels = len(intents)
        else:
            raise ValueError(
                "Looks like you passed a dataset name that isn't "
                "already supported by NeMo. Please make sure "
                "that you build the preprocessing method for it."
            )

        self.train_file = self.data_dir + '/train.tsv'

        for mode in ['train', 'test', 'eval', 'dev']:
            if not if_exist(self.data_dir, [f'{mode}.tsv']):
                logging.info(f'Stats calculation for {mode} mode is skipped as {mode}.tsv was not found.')
                continue

            input_file = f'{self.data_dir}/{mode}.tsv'
            with open(input_file, 'r') as f:
                input_lines = f.readlines()[1:]  # Skipping headers at index 0

            try:
                int(input_lines[0].strip().split()[-1])
            except ValueError:
                error_message = (
                    f'Stats calculation for {mode} mode is skipped.'
                    + f' No numerical labels found for {mode}.tsv in {dataset_name} dataset.'
                )

                if mode == eval_file_prefix:
                    error_message += f' Use a different dataset for {mode}.'
                    raise ValueError(error_message)

                logging.info(error_message)
                continue

            queries, raw_sentences = [], []
            for input_line in input_lines:
                parts = input_line.strip().split()
                raw_sentences.append(int(parts[-1]))
                queries.append(' '.join(parts[:-1]))

            infold = input_file[: input_file.rfind('/')]

            logging.info(f'Three most popular classes in {mode} dataset')
            total_sents, sent_label_freq = get_label_stats(raw_sentences, infold + f'/{mode}_sentence_stats.tsv')

            if mode == 'train':
                self.class_weights = calc_class_weights(sent_label_freq)
                logging.info(f'Class weights are - {self.class_weights}')

            logging.info(f'Total Sentences - {total_sents}')
            logging.info(f'Sentence class frequencies - {sent_label_freq}')
