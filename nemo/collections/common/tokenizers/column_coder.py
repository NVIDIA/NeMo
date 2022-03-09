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

from typing import Dict, List, Tuple
from cv2 import log
from pandas import Series
import numpy as np
import math
from nemo.utils import logging


class Code(object):

    def __init__(self, code_len: int, start_id: int):
        self.code_len = code_len
        self.start_id = start_id
        self.end_id = start_id

    def encode(self, item: str) -> List[int]:
        raise NotImplementedError()

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError()

    @property
    def code_range(self) -> List[Tuple[int, int]]:
        """
        get the vocab id range for each of the encoded tokens
        @returns [(min, max), (min, max), ...]
        """
        return [(self.start_id, self.end_id)]


class FloatCode(Code):

    def __init__(self, col_name: str, data_series: Series, code_len: int,
                 start_id: int, base: int = 100, fillall: bool = True):
        super().__init__(code_len, start_id)
        data_series = data_series.dropna()
        self.name = col_name
        self.mval = data_series.min()
        self.base = base
        # values are larger than zero
        # use log transformation to reduce the gap
        values = np.log(data_series - self.mval + 1.0)
        # assume base 10 numbers, can change the base of the values if
        # larger dictionary is needed
        digits = int(math.log(values.max(), base)) + 1
        # extra digits used for 'float' part of the number
        extra_digits = code_len - digits
        if extra_digits < 0:
            raise "need large length to code the nummber"
        # convert the float number into integer
        significant_val = (values * base**extra_digits).astype(int)
        self.extra_digits = extra_digits
        digits_id_to_item = [{} for _ in range(code_len)]
        digits_item_to_id = [{} for _ in range(code_len)]
        for i in range(code_len):
            id_to_item = digits_id_to_item[i]
            item_to_id = digits_item_to_id[i]
            v = (significant_val // base**i) % base
            if fillall:
                uniq_items = range(0, base)
            else:
                uniq_items = sorted(v.unique().tolist())
            for i in range(len(uniq_items)):
                item = str(uniq_items[i])
                item_to_id[item] = self.end_id
                id_to_item[self.end_id] = item
                self.end_id += 1
        self.end_id += 1  # add the N/A token
        self.digits_id_to_item = digits_id_to_item
        self.digits_item_to_id = digits_item_to_id
        self.NA_token = 'nan'
        codes = []
        ranges = self.code_range
        for i in ranges:
            codes.append(i[1]-1)
        self.NA_token_id = codes

    @property
    def code_range(self) -> List[Tuple[int, int]]:
        """
        get the vocab id range for each of the encoded tokens
        @returns [(min, max), (min, max), ...]
        """
        # first largest digits
        outputs = []
        c = 0
        for i in reversed(range(self.code_len)):
            ids = self.digits_id_to_item[i].keys()
            if c == 0:
                outputs.append(
                    (min(ids),
                     max(ids) + 2))  # the first token contains the N/A
            else:
                outputs.append((min(ids), max(ids)+1))
            c += 1
        return outputs

    def encode(self, item: str) -> List[int]:
        if item == self.NA_token:
            return self.NA_token_id
        val = float(item)
        values = np.log(val - self.mval + 1.0)
        val_int = (values * self.base**self.extra_digits).astype(int)
        digits = []
        for i in range(self.code_len):
            digit = (val_int // self.base**i) % self.base
            digits.append(str(digit))
        if (val_int // self.base**self.code_len) != 0:
            raise ValueError("not right length")
        codes = []
        for i in reversed(range(self.code_len)):
            digit_str = digits[i]
            if digit_str in self.digits_item_to_id[i]:
                codes.append(
                    self.digits_item_to_id[i][digit_str])
            else:
                # find the nearest encode id
                allowed_digits = np.array([int(d) for d in self.digits_item_to_id[i].keys()])
                near_id = np.argmin(np.abs(allowed_digits - int(digit_str)))
                digit_str = str(allowed_digits[near_id])
                codes.append(
                    self.digits_item_to_id[i][digit_str])
                logging.warning('out of domain num is encounterd, use nearest code')
        return codes

    def decode(self, ids: List[int]) -> str:
        if ids[0] == self.NA_token_id[0]:
            return self.NA_token
        v = 0
        for i in reversed(range(self.code_len)):
            digit = int(self.digits_id_to_item[i][ids[self.code_len - i - 1]])
            v += digit * self.base**i
        # v = int("".join(items))
        v = v / self.base**self.extra_digits
        accuracy = max(int(abs(np.log10(0.1 / self.base**self.extra_digits))), 1)
        v = np.exp(v) + self.mval - 1.0
        return f"{v:.{accuracy}f}"


class ColumnCode(Code):

    def __init__(self, col_name: str, data_series: Series, start_id: int):
        super().__init__(1, start_id)
        self.name = col_name
        uniq_items = data_series.unique().tolist()
        id_to_item = {}
        item_to_id = {}
        for i in range(len(uniq_items)):
            item = str(uniq_items[i])
            item_to_id[item] = self.end_id
            id_to_item[self.end_id] = item
            self.end_id += 1
        self.id_to_item = id_to_item
        self.item_to_id = item_to_id

    def encode(self, item) -> List[int]:
        return [self.item_to_id[item]]

    def decode(self, ids: List[int]) -> str:
        return self.id_to_item[ids[0]]


class ColumnCodes:

    def __init__(self):
        self.column_codes: Dict[str, Code] = {}
        self.columns = []
        self.sizes = []

    @property
    def vocab_size(self):
        return self.column_codes[self.columns[-1]].end_id

    def register(self, name: str, ccode: Code):
        self.columns.append(name)
        self.column_codes[name] = ccode
        self.sizes.append(ccode.code_len)

    def encode(self, col: str, item: str) -> List[int]:
        if col in self.column_codes:
            return self.column_codes[col].encode(item)
        else:
            raise ValueError(f"cannot encode {col} {item}")

    def decode(self, col: str, ids: List[int]) -> str:
        if col in self.column_codes:
            return self.column_codes[col].decode(ids)
        else:
            raise ValueError("cannot decode")

    def get_range(self, column_id: int) -> List[Tuple[int, int]]:
        return self.column_codes[self.columns[column_id]].code_range
