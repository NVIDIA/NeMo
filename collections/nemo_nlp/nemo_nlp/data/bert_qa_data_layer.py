# Copyright (c) 2019 NVIDIA Corporation

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *
from nemo.core import DeviceType
import torch
from .datasets import BertQuestionAnsweringDataset


class BertQuestionAnsweringDataLayer(DataLayerNM):
    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "input_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_type_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(TimeTag)
            }),
            "start_positions":
            NeuralType({0: AxisType(BatchTag)}),
            "end_positions":
            NeuralType({0: AxisType(BatchTag)}),
            "unique_ids":
            NeuralType({0: AxisType(BatchTag)})
        }

        return input_ports, output_ports

    def __init__(
            self, *,
            tokenizer,
            path_to_data,
            data_format,
            features_file_prefix,
            max_seq_length,
            is_training,
            max_query_length,
            local_rank,
            **kwargs
    ):
        DataLayerNM.__init__(self, **kwargs)

        self._device = torch.device(
            "cuda" if self.placement in [DeviceType.GPU, DeviceType.AllGpu]
            else "cpu"
        )

        self._dataset = BertQuestionAnsweringDataset(
            tokenizer=tokenizer,
            input_file=path_to_data,
            data_format=data_format,
            features_file_prefix=features_file_prefix,
            max_seq_length=max_seq_length,
            is_training=is_training,
            max_query_length=max_query_length,
            local_rank=local_rank)

    def calculate_exact_match_and_f1(self,
                                     unique_ids,
                                     start_logits,
                                     end_logits,
                                     n_best_size=20,
                                     max_answer_length=30,
                                     do_lower_case=False,
                                     version_2_with_negative=False,
                                     null_score_diff_thresold=0.0):
        exact_match, f1 = self._dataset.calculate_exact_match_and_f1(
            unique_ids, start_logits, end_logits, n_best_size,
            max_answer_length, do_lower_case, version_2_with_negative,
            null_score_diff_thresold)
        return exact_match, f1

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
