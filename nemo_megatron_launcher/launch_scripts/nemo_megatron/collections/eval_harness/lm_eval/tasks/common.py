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

import datasets

from ..base import Task


class HFTask(Task):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, cache_dir=""):
        self.data = None
        self.cache_dir = cache_dir
        super().__init__()

    def download(self):
        self.data = datasets.load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME, cache_dir=self.cache_dir)

    def has_training_docs(self):
        """Whether the task has a training set"""
        return True if "train" in self.data.keys() else False

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return True if "validation" in self.data.keys() else False

    def has_test_docs(self):
        """Whether the task has a test set"""
        return True if "test" in self.data.keys() else False

    def _convert_standard(self, doc):
        return doc

    def training_docs(self):
        # Cache training for faster few-shot.
        # If data is too large to fit in memory, override this method.
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(map(self._convert_standard, self.data["train"]))
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._convert_standard, self.data["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._convert_standard, self.data["test"])


def yesno(x):
    if x:
        return "yes"
    else:
        return "no"
