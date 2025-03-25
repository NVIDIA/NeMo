# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.llm.gpt.data.fine_tuning import HFFineTuningDataModule

class CNNDailyMailFineTuningDataModule(HFFineTuningDataModule):
    """A data module for fine-tuning on the CNN / Daily Mail dataset.

    This class inherits from the `HFFineTuningDataModule` class including arguments for init and these methods.
    """

    def _make_splits(self, dset):
        """Maps train/validation/test to standard split names."""
        save_splits = {
            "training": dset.get("train"),
            "validation": dset.get("validation"),
            "test": dset.get("test"),
        }
        return save_splits

    def _json_line_from_example(self, example):
        """Extract data for summarization task."""
        json_line = {
            "input": example["article"],
            "output": example["highlights"],
        }
        return json_line

    @property
    def dataset_name(self) -> str:
        return "cnn_dailymail"

    @property
    def hf_load_dataset_kwargs(self) -> dict:
        """Retrieve 1.0.0 version of the dataset."""
        kwargs = super().hf_load_dataset_kwargs | {
            "name": "1.0.0",
        }
        return kwargs
