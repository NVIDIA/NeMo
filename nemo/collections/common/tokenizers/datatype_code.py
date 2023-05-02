import math
from typing import List, Tuple, Union

import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, QuantileTransformer, RobustScaler

from code_spec import Code

# from nemo.utils import logging

__all__ = ["IntCode", "FloatCode", "CategoryCode"]


class NumericCode(Code):
    NA_token = 'nan'

    def __init__(self,
                 col_name: str,
                 code_len: int,
                 start_id: int,
                 base: int,
                 fill_all: bool = True,
                 has_nan: bool = True):
        super().__init__(col_name, code_len, start_id)
        self.fill_all = fill_all
        self.has_nan = has_nan
        self.base = base
        self.digits_id_to_item = None
        self.digits_item_to_id = None
        self.NA_token_id = None

    def transform_data_to_int_array(self, data_series: ndarray):
        """
        Some custom function that converts a numeric series to an integer array where the number of integral digits
        is the precision (ideally)

        Args:
            data_series:

        Returns:

        """
        # todo handle +/- inf as special tokens
        raise NotImplementedError()

    def compute_code(self, data_series: ndarray):
        """
        Calculates the mapping of input integers to tokens to token_ids. Calculated once on data.

        Args:
            data_series (ndarray): single column of integer data

        Returns:

        """
        significant_val = self.transform_data_to_int_array(
            np.unique(data_series))  # saves min of data_series and subtracts it out of data
        self.tokenize_numeric_data(significant_val)

    def tokenize_numeric_data(self, significant_val: ndarray):
        """

        Args:
            significant_val: integer array of unique values in data with the desired number of integral precision. The
            data are tokenized and added to the dictionary mapping values to token ids and token ids to values

        Returns: None

        """
        # tokenization scheme
        digits_id_to_item = [{} for _ in range(self.code_len)]
        digits_item_to_id = [{} for _ in range(self.code_len)]
        for i in range(self.code_len):
            id_to_item = digits_id_to_item[i]
            item_to_id = digits_item_to_id[i]
            # divides based on magnitude and computes modulus of result.
            # i.e. sig_val = 1003, base=10, i = 0 (4003//(10**0)) % 10 = 0 since 400 % 10 = 0
            v = (significant_val // self.base ** i) % self.base
            if self.fill_all:
                uniq_items = range(0, self.base)
            else:
                uniq_items = sorted(np.unique(v))
            for k in range(len(uniq_items)):
                item = str(uniq_items[k])
                item_to_id[item] = self.end_id
                id_to_item[self.end_id] = item
                self.end_id += 1
        self.digits_id_to_item = digits_id_to_item
        self.digits_item_to_id = digits_item_to_id
        if self.has_nan:
            self.end_id += 1  # add the N/A token
            codes = []
            ranges = self.code_range
            for i in ranges:
                codes.append(i[1] - 1)
            self.NA_token_id = codes


class IntCode(NumericCode):

    def __init__(self, col_name: str, code_len: int, start_id: int, fill_all: bool = True, base: int = 100,
                 has_nan: bool = True):

        super(IntCode, self).__init__(col_name, code_len, start_id, base, fill_all, has_nan)
        self.min_value = -np.inf
        # self.transform = None  # only used for FloatCode
        self.digits_id_to_item = None
        self.digits_item_to_id = None
        self.NA_token_id = None

    def compute_code(self, data_series: ndarray):
        """
        Calculates the mapping of input integers to tokens to token_ids. Calculated once on data.

        Args:
            data_series (ndarray): single column of integer data

        Returns:

        """
        # saves min of data_series and subtracts it out of data
        significant_val = self.transform_data_to_int_array(np.unique(data_series))
        self.tokenize_numeric_data(significant_val)

    def transform_data_to_int_array(self, array: ndarray):
        """
        Special function used by IntCode that performs some transformation on the val array to create an int array.
        In the simplest case, with an int array, we subtract out the minimum to make the min equal to zero.

        Saves the minimum value for use when training on new data or for transforming data back to the original space
        during generation.
        Args:
            array: an array (int array if used in IntCode)

        Returns:
            an integer array.

        """
        array = array.astype(int)
        self.min_value = array.min()
        if self.min_value == -np.inf:
            raise ValueError(
                'minimum value of array is negative infinity. Please map infinity to other value or consider using'
                'a special token to represent + or - infinity')
        return array - self.min_value

    def _transform_val_to_int(self, val: Union[int, float]) -> int:
        """
        Function used during encoding of integers. Not used in compute code
        Args:
            val (int or float):

        Returns:
        """
        return int(val) - self.min_value

    def _transform_int_to_val(self, decoded: Union[int, float]) -> int:
        """
        Function used during decoding process to convert integers back to original space. Not used in compute code
        Args:
            decoded (int or float):

        Returns: int with min of input data array added back in.

        """
        return decoded + self.min_value

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
                if self.has_nan:
                    outputs.append((min(ids), max(ids) + 2))  # the first token contains the N/A
                else:
                    outputs.append((min(ids), max(ids) + 1))  # non N/A
            else:
                outputs.append((min(ids), max(ids) + 1))
            c += 1
        return outputs

    def encode(self, item: str) -> List[int]:
        if self.has_nan and item == self.NA_token:
            return self.NA_token_id
        elif not self.has_nan and item == self.NA_token:
            raise ValueError(f"column {self.name} cannot handle nan, please set has_nan=True")
        val = float(item)
        val_int = self._transform_val_to_int(val)
        digits = []
        for i in range(self.code_len):
            digit = (val_int // self.base ** i) % self.base
            digits.append(str(digit))
        if (val_int // self.base ** self.code_len) != 0:
            raise ValueError("not right length")
        codes = []
        for i in reversed(range(self.code_len)):
            digit_str = digits[i]
            if digit_str in self.digits_item_to_id[i]:
                codes.append(self.digits_item_to_id[i][digit_str])
            else:
                # find the nearest encode id
                allowed_digits = np.array([int(d) for d in self.digits_item_to_id[i].keys()])
                near_id = np.argmin(np.abs(allowed_digits - int(digit_str)))
                digit_str = str(allowed_digits[near_id])
                codes.append(self.digits_item_to_id[i][digit_str])
                # logging.warning('out of domain num is encountered, use nearest code')
        return codes

    def decode(self, ids: List[int]) -> str:
        if self.has_nan and ids[0] == self.NA_token_id[0]:
            return self.NA_token
        v = 0
        for i in reversed(range(self.code_len)):
            digit = int(self.digits_id_to_item[i][ids[self.code_len - i - 1]])
            v += digit * self.base ** i
        v = self._transform_int_to_val(v)
        return str(v)


class FloatCode(NumericCode):
    AVAILABLE_TRANSFORMS = ["yeo-johnson", "quantile", "robust", "identity", "log1p"]  # best is not really a transform

    def __init__(
            self,
            col_name: str,
            code_len: int,
            start_id: int,
            fill_all: bool = True,
            base: int = 100,
            has_nan: bool = True,
            transform: str = 'quantile',
    ):
        super(FloatCode, self).__init__(col_name, code_len, start_id, base, fill_all, has_nan)
        self.min_value = -np.inf
        self.transform: str = transform
        self.scalar = None
        self.scalar = self.set_scalar_transform()

        self.extra_digits: int = 0
        self.digits_id_to_item = None
        self.digits_item_to_id = None
        self.NA_token_id = None

    def compute_code(self, data_series: ndarray):
        """
        Calculates the mapping of input floats to tokens to token_ids. Calculated once on data.


        digits_item_to_id is a list of dicts, where each dict takes a number and maps it to a token id
        digits_id_to_item is a list of dicts, where each dict takes a token id and maps it to a number

        The number of dictionaries equals the code length (or precision) desired.


        Args:
            data_series (ndarray): a single column of float data

        Returns:

        """
        if self.transform == 'best':
            self.transform: str = self.get_best_transform(data_series)
            self.scalar = self.set_scalar_transform()

        significant_val = self.transform_data_to_int_array(np.unique(data_series))
        self.tokenize_numeric_data(significant_val)

    def transform_data_to_int_array(self, array: ndarray) -> ndarray:
        """
        Takes an array of data and does a transformation (such as remove min value and take log1p, or just apply a
        PowerTransform). Then raise all the data to some number of precision digits and take the int of these.

        Args:
            array: a floating point array

        Returns: values, an integer array of the transformed data where the number of digits is the precision of the
        transformation

        """
        # this could be reparametrized into a "subtract_min_before_transform"
        if self.transform == 'log1p':
            self.min_value = array.min()
            array -= self.min_value
            values = self.scalar.fit_transform(array[:, None])[:, 0]
        else:
            values = self.scalar.fit_transform(array[:, None])[:, 0]
            self.min_value = values.min()
            values -= self.min_value

        if self.min_value == -np.inf:
            raise ValueError('minimum value of array is negative infinity. Please map infinity to other value or '
                             'consider using a special token to represent + or - infinity')

        # extra digits used for 'float' part of the number
        digits = int(math.log(values.max(), self.base)) + 1
        extra_digits: int = self.code_len - digits
        if extra_digits < 0:
            raise ValueError("need large length to code the number")
        self.extra_digits: int = extra_digits
        values = (values * self.base ** self.extra_digits).astype(int)
        return values

    def set_scalar_transform(self):
        """
        Runs through the available transforms and selects the transform with the smallest error based on binning the
        data. The 'best' transform option is calculated in another function, where min errors are chosen based on some
        simple data heuristics across the data distribution

        Returns: None

        """
        if self.transform == 'yeo-johnson':
            return PowerTransformer(standardize=True)
        elif self.transform == 'quantile':
            return QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
        elif self.transform == 'robust':
            return RobustScaler()
        elif self.transform is None or (isinstance(self.transform, str) and
                                        self.transform.lower() in ['none', '', 'identity']):
            # sklearn FunctionTransformer implements identity transformation by default.
            return FunctionTransformer(validate=True)
        elif self.transform == 'log1p':
            return FunctionTransformer(func=np.log1p,
                                       inverse_func=lambda x: np.exp(x) - 1,
                                       validate=True,
                                       check_inverse=True)
        elif self.transform != 'best':
            raise ValueError('Supported data transformations are "best", "yeo-johnson", "quantile", "robust", log1p, '
                             'and "identity"')

    def get_best_transform(self, data_series: ndarray) -> str:
        """
        Attempts to find the best transform for the given input data that minimizes the transformation error.
        Can use these results to override the best choice transform

        Args:
            data_series:

        Returns: best_transform (str)

        """
        num_bins = 31
        uniq_data = np.unique(data_series)  # saves on compute cost for repeated elements

        results = {'transform': [],
                   'mse': [],
                   'max_err': [],
                   'min_err': [],
                   'errors_across_series': [],
                   }
        for transform in self.AVAILABLE_TRANSFORMS:  # todo
            self.transform = transform
            self.set_scalar_transform()
            self.compute_code(uniq_data)

            error, error_pct, stats, distr_of_err_pcts, errs_throughout = self.calculate_transform_errors(data_series,
                                                                                                          bins=num_bins)
            mse, max_err, min_err = stats
            errors_across_series, bins = errs_throughout
            results['transform'].append(transform)
            results['mse'].append(mse)
            results['max_err'].append(max_err)
            results['min_err'].append(min_err)
            results['errors_across_series'].append(errors_across_series)
        errors_across_series = results.pop('errors_across_series')
        data = pd.concat([pd.DataFrame(results), pd.DataFrame(errors_across_series)], axis=1)
        # for now just use the errors across series and take the counts if it has the smallest error
        best_ranked_index = data.loc[:, range(0, num_bins)].apply(lambda x: [True if i == min(x) else False
                                                                             for i in x], axis=0).sum(axis=1).argmax()
        if hasattr(best_ranked_index, '__iter__'):  # in case of a tie-breaker
            best_ranked_index = best_ranked_index[0]
        best_transform = data.loc[best_ranked_index, 'transform']
        return best_transform

    def calculate_transform_errors(self, data_series: ndarray, bins: int) -> Tuple:
        """
        Take input data series, encode and then decode. Then calculate the error of the input data series and the
        decoded data series.

        Args:
            data_series: input data
            bins (int): number of bins for some simple errors across different bins of data

        Returns:
            error (ndarray) of exact input - decoded values
            error_pct (ndarray) the positive error percentages for each value of the input
            [mse, max_err, min_err] - [float, float, float]
            distr_of_err_pcts (ndarray) of errors at different percentiles/quantiles of the input
            [errors_across_series, bins_edges] - errors across the number of histogram bins and their corresponding bin
            edges in the data

        """
        str_data = data_series.astype(str)
        decoded = np.array([self.decode(self.encode(i)) for i in str_data]).astype(float)

        error = data_series - decoded
        error_pct = 100 * np.abs(error) / np.abs(data_series)  # careful of nan here
        mse = np.nanmean(error ** 2)
        max_err = np.nanmax(error)
        min_err = np.nanmin(error)
        # distribution of the error percentages0
        distr_of_err_pcts = [np.nanpercentile(error_pct, pct) for pct in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]]
        # distribution of the error across the range of the data_series, useful if there's a multimodal distribution
        bins_edges = np.histogram_bin_edges(data_series, bins=bins)

        errors_across_series = []
        for idx in range(len(bins_edges) - 1):
            error_subset = error_pct[(bins_edges[idx] < data_series) & (data_series < bins_edges[idx + 1])]
            if len(error_subset):
                error_subset = error_subset.max()
            else:  # error_subset == []
                error_subset = 0
            errors_across_series.append(error_subset)
        return error, error_pct, [mse, max_err, min_err], distr_of_err_pcts, [errors_across_series, bins_edges]

    def _transform_val_to_int(self, val: float) -> int:
        """
        After the compute code is run. This is part of the encode process (hence the underscore) for taking training or
        input data for inference, and transforming the value into an integer with the required precision.

        Args:
            val (float):

        Returns: integer

        """
        ret_array = False
        if isinstance(val, float):
            val = np.expand_dims(np.array(val), axis=0)
        else:
            ret_array = True
        if self.transform == 'log1p':
            values = self.scalar.transform(val[:, None] - self.min_value)[:, 0]
        else:
            values = self.scalar.transform(val[:, None])[:, 0] - self.min_value

        values = (values * self.base ** self.extra_digits).astype(int)
        if ret_array:
            return values
        output = values[0]
        return output

    def _transform_int_to_val(self, decoded: int) -> float:
        decoded = decoded / self.base ** self.extra_digits
        decoded = np.expand_dims(np.array(decoded), axis=0)
        if self.transform == 'log1p':
            val = self.scalar.inverse_transform(decoded[:, None])[0, 0] + self.min_value
        else:
            val = self.scalar.inverse_transform(decoded[:, None] + self.min_value)[0, 0]
        return val

    def encode(self, item: str) -> List[int]:
        if self.has_nan and item == self.NA_token:
            return self.NA_token_id
        elif not self.has_nan and item == self.NA_token:
            raise ValueError(f"column {self.name} cannot handle nan, please set has_nan=True")
        val = float(item)
        val_int = self._transform_val_to_int(val)
        digits = []
        for i in range(self.code_len):
            digit = (val_int // self.base ** i) % self.base
            digits.append(str(digit))
        if (val_int // self.base ** self.code_len) != 0:
            raise ValueError("not right length. too many digits")
        codes = []
        for i in reversed(range(self.code_len)):
            digit_str = digits[i]
            if digit_str in self.digits_item_to_id[i]:
                codes.append(self.digits_item_to_id[i][digit_str])
            else:
                # find the nearest encode id
                allowed_digits = np.array([int(d) for d in self.digits_item_to_id[i].keys()])
                near_id = np.argmin(np.abs(allowed_digits - int(digit_str)))
                digit_str = str(allowed_digits[near_id])
                codes.append(self.digits_item_to_id[i][digit_str])
                # logging.warning('out of domain num is encountered, use nearest code')
        return codes

    def decode(self, ids: List[int]) -> str:
        if self.has_nan and ids[0] == self.NA_token_id[0]:
            return self.NA_token
        v = 0
        for i in reversed(range(self.code_len)):
            digit = int(self.digits_id_to_item[i][ids[self.code_len - i - 1]])
            v += digit * self.base ** i
        v = self._transform_int_to_val(v)
        accuracy = max(int(abs(np.log10(0.1 / self.base ** (2 + self.extra_digits)))), 1)  # add extra precision
        return f"{v:.{accuracy}f}"


class CategoryCode(Code):
    def __init__(self, col_name: str, start_id: int):
        super().__init__(col_name, 1, start_id)
        self.id_to_item = {}
        self.item_to_id = {}

    def compute_code(self, data_series: ndarray):
        uniq_items: List = list(np.unique(data_series).astype(str))
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
