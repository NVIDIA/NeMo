# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
# =============================================================================
# Copyright (C) IBM Corporation 2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Tomasz Kornuta"

# This file contains code artifacts adapted from the original implementation:
# https://github.com/IBM/pytorchpipe/blob/develop/ptp/components/models/language/sentence_embeddings.py

import csv
import zipfile
from dataclasses import MISSING, dataclass, field
from os.path import dirname, exists, expanduser, join
from typing import List

import numpy as np
import torch
import tqdm

import nemo  # to get version.
from nemo.core.classes import NeuralModule
from nemo.utils import logging
from nemo.utils.cloud import maybe_download_from_cloud


@dataclass
class SentenceEmbeddingsConfig:
    word_mappings_filepath: str = MISSING
    embeddings_size: int = MISSING
    additional_tokens: List[str] = field(default_factory=list)
    eos_token: bool = False
    fixed_padding_length: int = -1
    pretrained_embeddings: str = ''
    skip_unknown_words: bool = False
    _target_: str = "nemo.collections.vis.modules.SentenceEmbeddings"


class SentenceEmbeddings(NeuralModule):
    """
    Module responsible for embedding of whole sentences:
        - uses provided vocabulary to first transform words into indices (one by one)
        - transform indices to dense vectors.

    Optionally, it can load the pretrained word embeddings (GloVe).
    """

    def __init__(
        self,
        word_mappings_filepath: str,
        embeddings_size: int,
        additional_tokens: List[str] = ['<PAD>'],
        eos_token: bool = False,
        fixed_padding_length: int = -1,
        pretrained_embeddings: str = '',
        skip_unknown_words: bool = False,
    ):
        """
        Creates the module.

        Args:
            word_mappings_filepath: File containing word-id mappings.

            embeddings_size: Size of embeddings.

            additional_tokens: List of additional tokens that will be added to vocabulary.
                This list can be extended, but <PAD> and <EOS> are special tokens.
                <PAD> is ALWAYS used for padding shorter sequences.
            eos_token: Enable <EOS> (end of sequence) token (DEFAULT: False)

            fixed_padding_length: Fixed padding length
                -1  -> For each batch, automatically pad to the length of the longest sequence in the batch
                (i.e. varying from batch to batch)
                > 0 -> Pad each pad to the chosen length (fixed for all batches)

            pretrained_embeddings: File containing pretrained embeddings.
                Empty means that no embeddings will be loaded.
                Options:
                '' | glove.6B.50d.txt | glove.6B.100d.txt | glove.6B.200d.txt | glove.6B.300d.txt |
                glove.42B.300d.txt | glove.840B.300d.txt | glove.twitter.27B.txt | mimic.fastText.no_clean.300d.pickled

            skip_unknown_words: Skips words out of dictionary (DEFAULT: False)
        """
        # Call the base class constructor.
        super().__init__()

        # Set embeddings size.
        self._embeddings_size = embeddings_size

        # Load word embeddings.
        self._word_to_ix = self.load_word_mappings_from_csv_file(word_mappings_filepath)

        # Insert <PAD> to word mappings - if required.
        if '<PAD>' not in self._word_to_ix:
            self._word_to_ix['<PAD>'] = len(self._word_to_ix)
        # Get index of <PAD> from vocabulary.
        self._pad_index = self._word_to_ix['<PAD>']

        # Handle <EOS> token.
        if eos_token:
            if '<EOS>' not in self._word_to_ix:
                # Insert <EOS> to word mappings.
                self._word_to_ix['<EOS>'] = len(self._word_to_ix)

        # Create the embeddings layer.
        logging.info(
            "Initializing embeddings layer with vocabulary size = {} and embeddings size = {}".format(
                len(self._word_to_ix), self._embeddings_size
            )
        )
        self._embeddings = torch.nn.Embedding(
            len(self._word_to_ix), self._embeddings_size, padding_idx=self._pad_index
        )

        # Forces padding to a fixed length.
        self._fixed_padding_length = fixed_padding_length

        # Skips words out of dictionary.
        self._skip_unknown_words = skip_unknown_words

        # Load the embeddings first.
        if pretrained_embeddings != '':
            emb_vectors = self.load_pretrained_embeddings(pretrained_embeddings)
            # Set weights
            self._embeddings.weight = torch.nn.Parameter(emb_vectors)

    def load_word_mappings_from_csv_file(self, filepath):
        """
        Loads (word:index) mappings from csv file.
        .. warning::
                There is an assumption that file will contain key,value pairs (no content checking for now!)

        Args:
            filepath: Path an name of file with encodings (absolute path + filename).

        Returns:
            Dictionary with word:index.
        """
        filepath = expanduser(filepath)

        if not exists(filepath):
            logging.error("Cannot load word mappings from '{}' because the file does not exist".format(filepath))

        word_to_ix = {}
        sniffer = csv.Sniffer()
        with open(filepath, mode='rt') as csvfile:
            # Check the presence of the header.
            first_bytes = str(csvfile.read(256))
            has_header = sniffer.has_header(first_bytes)
            # Rewind.
            csvfile.seek(0)
            reader = csv.reader(csvfile)
            # Skip the header row.
            if has_header:
                next(reader)
            # Read the remaining rows.
            for row in reader:
                if len(row) == 2:
                    word_to_ix[row[0]] = int(row[1])

        logging.info("Loaded mappings of size {}".format(len(word_to_ix)))
        return word_to_ix

    def load_pretrained_embeddings(self, embeddings_name, refresh_cache=False):
        """
        Creates embedding vector for words from the provided (word:index) mappings (dictionary).
        Loads the pretrained embeddings from the GloVe project - for the words found in the dictionary.
        For words out of dictionary initializes random vectors.

        Args:
            embeddings_name: Name of file containing embeddings. Available embeddings:
                - glove.6B.50d.txt
                - glove.6B.100d.txt
                - glove.6B.200d.txt
                - glove.6B.300d.txt
                - glove.42B.300d.txt
                - glove.840B.300d.txt
                - glove.twitter.27B.txt

        Returns:
            Torch tensor with loaded (or random) vectors.
        """
        # https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
        # http://ronny.rest/blog/post_2017_08_04_glove/

        # Check the presence of the file.
        # Available options.
        # https://nlp.stanford.edu/projects/glove/
        pretrained_embeddings_urls = {}
        pretrained_embeddings_urls["glove.6B.50d.txt"] = ("http://nlp.stanford.edu/data/", "glove.6B.zip")
        pretrained_embeddings_urls["glove.6B.100d.txt"] = ("http://nlp.stanford.edu/data/", "glove.6B.zip")
        pretrained_embeddings_urls["glove.6B.200d.txt"] = ("http://nlp.stanford.edu/data/", "glove.6B.zip")
        pretrained_embeddings_urls["glove.6B.300d.txt"] = ("http://nlp.stanford.edu/data/", "glove.6B.zip")
        pretrained_embeddings_urls["glove.42B.300d.txt"] = ("http://nlp.stanford.edu/data/", "glove.42B.300d.zip")
        pretrained_embeddings_urls["glove.840B.300d.txt"] = ("http://nlp.stanford.edu/data/", "glove.840B.300d.zip")
        pretrained_embeddings_urls["glove.twitter.27B.txt"] = (
            "http://nlp.stanford.edu/data/",
            "glove.twitter.27B.zip",
        )

        # Check selection.
        if embeddings_name not in pretrained_embeddings_urls.keys():
            logging.error(
                "Cannot load the indicated pretrained embeddings (current '{}' must be one of {})".format(
                    embeddings_name, pretrained_embeddings_urls.keys()
                )
            )
            exit(1)

        # Get /url.
        (url, filename) = pretrained_embeddings_urls[embeddings_name]
        cache_subfolder = f"NEMO_{nemo.__version__}"
        # If file exists on cache_folder/subfolder, it will be re-used, unless refresh_cache is True.
        zip_filepath = maybe_download_from_cloud(
            url=url, filename=filename, subfolder=cache_subfolder, refresh_cache=refresh_cache
        )
        folder = dirname(zip_filepath)

        # Check if file exists.
        if not exists(join(folder, embeddings_name)):
            # Extract data from zip.
            logging.info("Extracting data from '{}'".format(zip_filepath))
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(folder)

        num_loaded_embs = 0
        # Set random embeddings for words "out of vocabulary".
        # embeddings = np.zeros((len(word_to_ix), embeddings_size))
        embeddings = np.random.normal(scale=0.6, size=(len(self._word_to_ix), self._embeddings_size))

        # Get number of lines/vectors.
        num_lines = sum([1 for line in open(join(folder, embeddings_name))])
        t = tqdm.tqdm(total=num_lines)

        with open(join(folder, embeddings_name)) as f:
            # Parse file and cherry pick the vectors that fit our vocabulary.
            for line in f.readlines():
                values = line.split()
                if len(values) > self._embeddings_size + 1:
                    # print(len(values))
                    # Case: two (or more) words!
                    num_words = len(values) - self._embeddings_size
                    words = values[0:num_words]
                    word = ' '.join(words)
                    # print(word)
                    # Get remaining vector.
                    vector = np.array(values[num_words:], dtype='float32')
                else:
                    # Get word.
                    word = values[0]
                    # Get remaining vector.
                    vector = np.array(values[1:], dtype='float32')
                # Get index.
                index = self._word_to_ix.get(word)
                if index:
                    assert (
                        len(vector) == self._embeddings_size
                    ), "Embeddings size must be equal to the size of pretrained embeddings!"
                    # Ok, set vector.
                    embeddings[index] = vector
                    # Increment counter.
                    num_loaded_embs += 1
                t.update()
            t.close()

        logging.info(
            "Loaded {} pretrained embeddings for vocabulary of size {} from {}".format(
                num_loaded_embs, len(self._word_to_ix), embeddings_name
            )
        )

        # Return matrix with embeddings.
        return torch.from_numpy(embeddings).float()

    def pad_trunc_list(self, l: list, length: int, padding_value=0, eos_value=None):
        """
        Will apply padding / clipping to list to meet requested length.
        Works on the list in-place.
        :param l: List to manipulate
        :param length: Target length
        :param padding_value: Value to fill when padding. Default is int(0).
        :return: None
        """
        if len(l) < length:
            if eos_value is not None:
                l.append(eos_value)
            l.extend([padding_value] * (length - len(l)))

        elif len(l) > length:
            # print("pad_trunc_list to cat!: {}".format(len(l)))
            # exit(1)
            del l[length:]
            if eos_value is not None:
                l[length - 1] = eos_value

    # @typecheck()
    def forward(self, batch):
        """
        Performs the forward step.

        Args:
            batch: Batch of tokenized sentences

        Returns:
            Batch of embeddings.
        """

        indices_list = []
        # Process samples 1 by one.
        for sample in batch:
            assert isinstance(sample, (list,)), 'This embedder requires input sample to contain a list of words'
            # Process list.
            output_sample = []
            # Encode sample (list of words)
            for token in sample:
                # Skip if word is unknown.
                if self._skip_unknown_words and token not in self._word_to_ix:
                    continue
                # Get index.
                output_index = self._word_to_ix[token]
                # Add index to outputs.
                output_sample.append(output_index)

            # Apply fixed padding to all sequences if requested
            # Otherwise let torch.nn.utils.rnn.pad_sequence handle it and choose a dynamic padding
            if self._fixed_padding_length > 0:
                pad_trunc_list(output_sample, self.fixed_padding, padding_value=self._pad_index)

            tensor = torch.LongTensor(output_sample)
            # Move to cuda if required.
            if next(self._embeddings.parameters()).is_cuda:
                tensor = tensor.cuda()
            # Add tensor to list.
            indices_list.append(tensor)

        # Padd indices using pad index retrieved from vocabulary.
        padded_indices = torch.nn.utils.rnn.pad_sequence(indices_list, batch_first=True, padding_value=self._pad_index)
        # Embedd indices.
        embedds = self._embeddings(padded_indices)

        return embedds

    def save_to(self, save_path: str):
        """Not implemented yet.
           Serialize model.

        Args:
            save_path (str): path to save serialization.
        """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """Not implemented yet.
            Restore module from serialization.

        Args:
            restore_path (str): path to serialization
        """
        pass
