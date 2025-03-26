import json
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import re
import numpy as np
from datasets import load_dataset

from nemo.collections.llm.gpt.data.core import get_dataset_root
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs

BLOCK_COMMON="You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions.\n    <</SYS>>\n\n    Implement the Verilog module based on the following block level summaries. Assume that signals are positive clock/clk edge triggered unless otherwise stated.\nHere are block level summaries:\n\nblock_0:"
DETAILED_COMMON="You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions.\n    <</SYS>>\n\n    Implement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated."
HIGH_LEVEL_COMMON="You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions.\n    <</SYS>>\n\n    Implement the Verilog module based on the following description. Assume that signals are positive clock/clk edge triggered unless otherwise stated."

## subclass the finetuning data module to create our own verilog data
class VerilogDataModule(FineTuningDataModule, IOMixin):
    def __init__(
        self,
        seq_length: int = 1024,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 2,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        force_redownload: bool = False,
        delete_raw: bool = True,
        seed: int = 12,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        packed_sequence_specs: Optional["PackedSequenceSpecs"] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.force_redownload = force_redownload
        self.delete_raw = delete_raw

        super().__init__(
            # where you save the train, validation, test.jsonl in nemo format
            dataset_root=get_dataset_root("//workspace/data/verilog"),
            seq_length=seq_length,
            tokenizer=tokenizer,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            seed=seed,
            memmap_workers=memmap_workers,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            packed_sequence_specs=packed_sequence_specs,
        )

    def find_common_substrings(strings):
        common_substrings = set(strings[0])
        for string in strings[1:]:
            common_substrings &= set(re.findall(r'\w+', string))
        return common_substrings

    # override the base class function for data handling logic
    def prepare_data(self) -> None:
        # if train file is specified, no need to do anything
        if not self.train_path.exists() or self.force_redownload:
            dset = self._download_data()
            self._preprocess_and_split_data(dset)
        super().prepare_data()
    
    def _download_data(self):
        logging.info(f"Downloading {self.__class__.__name__}...")
        return load_dataset(
            "GaTech-EIC/MG-Verilog",
            cache_dir=str(self.dataset_root),
            download_mode="force_redownload" if self.force_redownload else None,
        )

    def _preprocess_and_split_data(self, dset, train_ratio: float=0.80, val_ratio: float=0.15):
        logging.info(f"Preprocessing {self.__class__.__name__} to jsonl format and splitting...")

        test_ratio = 1 - train_ratio - val_ratio
        save_splits = {}
        dataset = dset.get('train')
        split_dataset = dataset.train_test_split(test_size=val_ratio + test_ratio, seed=self.seed)
        split_dataset2 = split_dataset['test'].train_test_split(
            test_size=test_ratio / (val_ratio + test_ratio), seed=self.seed
        )
        save_splits['training'] = split_dataset['train']
        save_splits['validation'] = split_dataset2['train']
        save_splits['test'] = split_dataset2['test']

        for split_name, dataset in save_splits.items():
            output_file_high_level = self.dataset_root / f"{split_name}.jsonl"
            with output_file_high_level.open("w", encoding="utf-8") as :f
                for example in dataset:
                    code = example["code"].strip()
                    description = example["description"]
                    high_level_global_summary = description['high_level_global_summary']
                    high_level_global_summary = high_level_global_summary.replace(HIGH_LEVEL_COMMON, "")
                    f.write(json.dumps({"input": high_level_global_summary, "output": code}) + "\n")
            logging.info(f"{split_name} split saved to {output_file_high_level}")

        if self.delete_raw:
            for p in self.dataset_root.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                elif '.jsonl' not in str(p.name):
                    p.unlink()
