# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_griffin_model import MegatronGriffinModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel

from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
)
from nemo.utils import logging

try:
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    import logging


    HAVE_LDDL = True
except (ImportError, ModuleNotFoundError):
    HAVE_LDDL = False

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False

from functools import partial

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset, GPTSFTPackedDataset
from nemo.collections.nlp.data.language_modeling.megatron.griffin_sft_dataset import GriffinSFTDataset


from nemo.utils import logging

try:
    from apex.transformer.pipeline_parallel.utils import (
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
from nemo.utils import AppState, logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ['MegatronGriffinSFTModel']


class MegatronGriffinSFTModel(MegatronGriffinModel, MegatronGPTSFTModel):
    """
    Megatron Griffin Supervised Fine-Tuning
    """
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(cfg, trainer=trainer)
        self.sep_id = cfg.get('sep_id', 49704)
        if hasattr(self.cfg.data, "validation_ds"):
            self.val_metric, self.val_metric_name = self.setup_metric(self.cfg.data.validation_ds)
            self.val_metric = torch.nn.ModuleList(self.val_metric) if self.val_metric is not None else None
            # Used other keys from metadata to calulate metrics
            if hasattr(self.cfg.data.validation_ds, "metric"):
                self.val_metric_label_key = self.cfg.data.validation_ds.metric.get('label_key', 'labels')

        if hasattr(self.cfg.data, "test_ds"):
            self.test_metric, self.test_metric_name = self.setup_metric(self.cfg.data.test_ds)
            self.test_metric = torch.nn.ModuleList(self.test_metric) if self.test_metric is not None else None
            # Used other keys from metadata to calulate metrics
            if hasattr(self.cfg.data.test_ds, "metric"):
                self.test_metric_label_key = self.cfg.data.test_ds.metric.get('label_key', 'labels')

        # Set the profile start and end steps in the unit of global batach
        if hasattr(self, '_nsys_profile_enabled'):
            self._nsys_profile_start_step = self.cfg.nsys_profile.get('start_step', 0)
            self._nsys_profile_end_step = self.cfg.nsys_profile.get('end_step', 0)

        # self._reset_activation_checkpointing_args()
        self.use_peft = False
        self.virtual_tokens = 0
        self.init_global_step = 0

    def _build_dataset(self, data_cfg, is_train=True):
        packed_sequence = data_cfg.get("packed_sequence", False)
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_list_config = isinstance(data_cfg.file_names, ListConfig)
        if not is_list_config:
            raise ValueError(f"SFT train/validation datasets must be provided as a list of individual JSONL files.")

        if is_train:
            # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
            # that is of the format [weight1,file_name1,weight2,file_name2,...]
            if data_cfg.concat_sampling_probabilities is None or not isinstance(
                data_cfg.concat_sampling_probabilities, ListConfig
            ):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be a ListConfig with the same number of files in file_names."
                        f"Found: {data_cfg.concat_sampling_probabilities}"
                    )
                )

            if len(data_cfg.get('concat_sampling_probabilities', None)) != len(data_cfg.file_names):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be of the same size as file_names.",
                        f"Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(data_cfg.file_names)}",
                    )
                )

            data_prefix = []
            for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
                data_prefix.append(weight)
                data_prefix.append(prefix)

            if self.trainer.max_steps is None or self.trainer.max_steps <= 0:
                raise ValueError(
                    f'Trainer max_steps must be set to a positive integer. Found {self.trainer.max_steps}'
                )
            num_train_samples = [self.trainer.max_steps * data_cfg.global_batch_size]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
        else:
            num_train_samples_per_dataset = [[None]] * len(data_cfg.file_names)

        # Check dataset max_seq_legnth and max_position_embeddings size
        if (
            self.cfg.get('position_embedding_type', None) in [None, 'learned_absolute']
            and data_cfg.max_seq_length > self.cfg.max_position_embeddings
        ):
            logging.warning(
                f"Set dataset max_seq_length to max_position_embeddings {self.cfg.max_position_embeddings} if using learned_absolute position embedding"
            )
            data_cfg.max_seq_length = self.cfg.max_position_embeddings

        # TE requires that the first input dim is divisible by 8 and the second by 16 for fp8
        # When using sequence parallel, sequence will further be split by TP size
        pad_seq_length_to_mult = (
            8 * self.cfg.get('tensor_model_parallel_size', 1) if self.cfg.get('sequence_parallel', False) else 16
        )

        dataset_kwargs = {}
        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            if self.cfg.data.get("chat", False):
                dataset_cls = GPTSFTChatDataset
            elif packed_sequence:
                dataset_cls = GPTSFTPackedDataset
                dataset_kwargs = {'return_cu_seqlen': data_cfg.get("packed_sequence_return_cu_seqlen", True)}
                assert data_cfg.micro_batch_size == 1, "Micro batch size must be 1 if using packed sequence"
            else:
                dataset_cls = GriffinSFTDataset

            # TODO(akoumparouli): MCore assumes/requires equal length input sequences.
            if not data_cfg.get('pad_to_max_length', False) and self.cfg.get('expert_model_parallel_size', 1) > 1:
                raise ValueError('Expert parallelism requires pad_to_max_length')

            dataset = dataset_cls(
                file_path=file_path,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                pad_seq_length_to_mult=pad_seq_length_to_mult,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                add_sep=data_cfg.get('add_sep', False),
                sep_id=self.sep_id,
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
                label_key=data_cfg.get('label_key', 'answer'),
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                truncation_field=data_cfg.get('truncation_field', 'text'),
                pad_to_max_length=data_cfg.get('pad_to_max_length', False),
                index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                prompt_template=data_cfg.get('prompt_template', None),
                virtual_tokens=self.virtual_tokens,
                tokens_to_generate=data_cfg.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                memmap_workers=data_cfg.get(
                    'memmap_workers', None
                ),  # used to set num. of workers to create the memmap index files
                hf_dataset=data_cfg.get(
                    'hf_dataset', False
                ),  # Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
                truncation_method=data_cfg.get(
                    'truncation_method', 'right'
                ),  # used to choose truncation method. Options: ['random', 'left', 'right']
                special_tokens=self.cfg.data.get(
                    'chat_prompt_tokens', None
                ),  # special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
                is_test=not is_train,
                **dataset_kwargs,
            )
            datasets.append(dataset)
        if is_train:
            if packed_sequence:
                num_train_samples_after_blend = sum(len(dataset) for dataset in datasets)
            dataset = BlendableDataset(
                datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            return datasets
        
    def on_validation_epoch_start(self):
        # self._reset_activation_checkpointing_args()
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.validation_ds.global_batch_size,
            micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        # self._reset_activation_checkpointing_args()
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.test_ds.global_batch_size,
            micro_batch_size=self.cfg.data.test_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_test_epoch_start()

    def on_predict_epoch_start(self):
        return self.on_test_epoch_start()

    def on_test_epoch_end(self):
        _ = self.inference_epoch_end(self.test_step_outputs, 'test', self.cfg.data.test_ds)
        # Commenting as on_test_epoch_end was a no-op in PTL 1.9
        # return super().on_test_epoch_end()

    def on_validation_epoch_end(self):
        _ = self.inference_epoch_end(self.validation_step_outputs, 'validation', self.cfg.data.validation_ds)
        # Commenting as on_validation_epoch_end was a no-op in PTL 1.9
        # return super().on_validation_epoch_end()

    def on_train_epoch_start(self) -> None:
        # Same logic as validation epoch end, but this may be need if there is no validation sanity check to trigger on_validation_epoch_end()
        self.on_validation_epoch_end()
        return super().on_train_epoch_start()
    