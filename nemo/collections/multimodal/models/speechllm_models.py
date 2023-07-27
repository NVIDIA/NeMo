# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import os
from typing import Optional, Union, Dict

import torch
from omegaconf.dictconfig import DictConfig
from nemo.utils import logging, model_utils
from nemo.collections.asr.data.audio_to_text_dali import (
    DALIOutputs,
)
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    SamplingParam,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import (
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


__all__ = ["ModularizedSpeechGPTModel"]


class ModularizedSpeechGPTModel(MegatronGPTPromptLearningModel):
    """Modularized speech GPT model."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg
        super().__init__(cfg, trainer)
        self.init_perception_model(cfg, trainer)

    def init_perception_model(self, cfg: DictConfig, trainer: Trainer):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        if not isinstance(cfg, DictConfig):
            raise ValueError("cfg must be an OmegaConf DictConfig")

        self.perception = ModularizedSpeechGPTModel.from_config_dict(
            self.cfg.perception
        )
        # TODO(zhehuai): load pretrained perception model weights

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size
        fixed_prompt_prefix_str = cfg.get("fixed_prompt_prefix", None)
        if fixed_prompt_prefix_str is not None:
            self.fixed_prompt_prefix = self.tokenizer.text_to_ids(
                fixed_prompt_prefix_str
            )
        else:
            self.fixed_prompt_prefix = None

    # follow MegatronGPTPromptLearningModel for GPT model init
    def init_model(self, cfg: DictConfig, trainer: Trainer):
        super().init_model(cfg, trainer)
        # gpt code handle the setup of the tokenizer
        # disable text prompt tuning specifics
        self.existing_tasks = None
        self.new_tasks = None
        self.virtual_prompt_style = None
        self.word_embeddings = (
            self.frozen_model.model.language_model.embedding.word_embeddings
        )
        self.pseudo_tokens = None
        self.pseudo_token_ids = None
        self.pseudo_token_ids_start = None
        self.virtual_prompt_source = None
        self.prompt_encoder = None
        # self.frozen_model is frozen by setup_optimizer_param_groups

    def state_dict(self):
        """
        TODO(zhehuai): Custom state dict.
        """
        state_dict_ = {}

        if self.first_stage_of_pipeline():
            pass

        return state_dict_

    def load_task_templates(self, task_templates):
        # TODO(zhehuai): support task template to support complexer SFT format
        self.task_templates = {}
        self.task_id_num_to_name = {}
        self.max_virtual_tokens = 0

    def get_text_batch_from_audio(self, audio_batch):
        _, _, transcript, transcript_len = audio_batch
        # TODO(zhehuai) Add BOS/EOS if desired, adds EOS by default
        labels = transcript[:, 1:].contiguous()
        input_ids = transcript[:, :-1].contiguous()
        input_length = transcript_len - 1

        b = labels.shape[0]
        max_len = labels.shape[1]
        # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
        loss_mask = torch.arange(max_len).expand(b, max_len) < input_length.unsqueeze(1)
        loss_mask = loss_mask.float()
        return input_ids, input_length, labels, loss_mask

    def prepare_llm_input(self, input_embeds, input_length):
        b = input_embeds.shape[0]
        max_len = input_embeds.shape[1]

        # Using causal attention mask for whole input
        # TODO(zhehuai): use prefixlm instead for the audio embeddings
        attention_mask = torch.tril(torch.ones((b, max_len, max_len))).view(
            b, 1, max_len, max_len
        )
        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5
        position_ids = build_position_ids(input_embeds[:, :, 0])

        # Add position embeddings
        if hasattr(
            self.frozen_model.model.language_model.embedding, "position_embeddings"
        ):
            position_embeddings = (
                self.frozen_model.model.language_model.embedding.position_embeddings(
                    position_ids
                )
            )
            input_embeds = input_embeds + position_embeddings
        else:
            input_embeds = input_embeds
        encoder_input = input_embeds.transpose(0, 1).contiguous()
        if self.cfg.get("sequence_parallel", False):
            encoder_input = (
                tensor_parallel.mappings.scatter_to_sequence_parallel_region(
                    encoder_input
                )
            )

        return encoder_input, attention_mask

    def forward(
        self,
        audio_batch,
        inference=True,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
    ):
        """Forward pass of the model.

        We first prepend a fixed text instruction that briefly describes the
        task to the audio embeddings. Then we prepend audio embeddings to
        the label text tokens as the LLM input.
        TODO(zhehuai): read text instruction from the SFT dataset, set loss_mask
          accordingly, following pad_batch_and_build_loss_mask.
        """

        # concat the text embeddings and the audio embeddings together to form the input embeddings
        def _concat_embs(embs1, emb1_lens, embs2, emb2_lens):
            concat_emb = []
            concat_len = []
            for emb1, emb1_len, emb2, emb2_len in zip(
                embs1, emb1_lens, embs2, emb2_lens
            ):
                new_len = emb1_len + emb2_len
                new_emb = torch.concat([emb1[:emb1_len], emb2[:emb2_len]], axis=0)
                padded_new_emb = torch.zeros(
                    emb1.shape[0] + emb2.shape[0], emb1.shape[-1]
                )
                padded_new_emb[:new_len, ...] = new_emb
                concat_emb.append(padded_new_emb)
                concat_len.append(new_len)
            concat_emb = torch.stack(concat_emb, dim=0)
            concat_len = torch.stack(concat_len, dim=0)
            return concat_emb, concat_len

        def _shift_labels_by_emb_len(
            labels, label_lens, emb_lens, max_len, pad_token=0
        ):
            shifted_labels = []
            for label, label_len, emb_len in zip(labels, label_lens, emb_lens):
                shifted_label = torch.full([max_len], pad_token)
                shifted_label[emb_len : emb_len + label_len] = label[:label_len]
                shifted_labels.append(shifted_label)
            shifted_labels = torch.stack(shifted_labels, dim=0)
            return shifted_labels

        signal, signal_len, _, _ = audio_batch

        # forward() only performs encoder forward
        if isinstance(audio_batch, DALIOutputs) and audio_batch.has_processed_signal:
            (
                input_signal,
                input_signal_length,
                processed_signal,
                processed_signal_length,
            ) = (None, None, signal, signal_len)
        else:
            (
                input_signal,
                input_signal_length,
                processed_signal,
                processed_signal_length,
            ) = (signal, signal_len, None, None)

        input_ids, input_length, labels, loss_mask = self.get_text_batch_from_audio(
            audio_batch
        )

        if not self.frozen_model.model.pre_process():
            raise ValueError("Model does not have pre_process method defined.")

        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )
        if self.fixed_prompt_prefix is not None:
            fixed_prompt_prefix = self.fixed_prompt_prefix.expand(encoded.shape[0], -1)
            prompt_prefix = self.word_embeddings(fixed_prompt_prefix)
            encoded = torch.cat([prompt_prefix, encoded], dim=1)
            encoded_len += fixed_prompt_prefix.shape[1]
        # [b, t, c]
        input_embeds = self.word_embeddings(input_ids)
        encoder_input, encoder_length = _concat_embs(
            encoded, encoded_len, input_embeds, input_length
        )
        labels = _shift_labels_by_emb_len(
            labels, input_length, encoded_len, encoder_input.shape[1], pad_token=0
        )
        # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
        loss_mask = _shift_labels_by_emb_len(
            loss_mask, input_length, encoded_len, encoder_input.shape[1], pad_token=0
        )
        encoder_input, attention_mask = self.prepare_llm_input(
            encoder_input, encoder_length
        )
        encoder_length = encoder_length
        output = self.frozen_model.model(
            input_ids=None,
            position_ids=None,
            encoder_input=encoder_input,
            attention_mask=attention_mask,
            labels=labels,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
        )

        return output, loss_mask

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            batch = [x.cuda(non_blocking=True) for x in batch]
            output_tensor, loss_mask = model(batch, inference=False)

            if isinstance(output_tensor, tuple):
                output_tensor, _ = output_tensor

            def loss_func(output_tensor):
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {"avg": reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch
        loss_mean = self.fwd_bwd_step(
            itertools.chain([batch]), None, forward_only=False
        )
        self.allreduce_gradients()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16 and hasattr(
            self.trainer.precision_plugin.scaler, "_scale"
        ):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log("loss_scale", loss_scale, batch_size=1)

        self.log(
            "reduced_train_loss",
            loss_mean,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("lr", lr, rank_zero_only=True, batch_size=1)
        self.log(
            "global_step",
            self.trainer.global_step,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )
        return loss_mean

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO(zhehuai) support infernece
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), None, forward_only=True)
        if loss_mean.item == 0.0:
            loss_mean = []
        return {"loss": loss_mean}

    # dataset configuration
    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToBPEDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config["shuffle"]
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], "collate_fn"):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config["batch_size"],
            collate_fn=collate_fn,
            drop_last=config.get("drop_last", False),
            shuffle=shuffle,
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False),
        )

    def _setup_transcribe_dataloader(
        self, config: Dict
    ) -> "torch.utils.data.DataLoader":
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if "manifest_filepath" in config:
            manifest_filepath = config["manifest_filepath"]
            batch_size = config["batch_size"]
        else:
            manifest_filepath = os.path.join(config["temp_dir"], "manifest.json")
            batch_size = min(config["batch_size"], len(config["paths2audio_files"]))

        dl_config = {
            "manifest_filepath": manifest_filepath,
            "sample_rate": self.preprocessor._sample_rate,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": config.get(
                "num_workers", min(batch_size, os.cpu_count() - 1)
            ),
            "pin_memory": True,
            "channel_selector": config.get("channel_selector", None),
            "use_start_end_token": self.cfg.validation_ds.get(
                "use_start_end_token", False
            ),
        }

        if config.get("augmentor"):
            dl_config["augmentor"] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(
            config=DictConfig(dl_config)
        )
        return temporary_datalayer

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        TODO(zhehuai): support unpaired data and the mixing of paired and unpaired data.
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if "shuffle" not in train_data_config:
            train_data_config["shuffle"] = True

        # preserve config
        self._update_dataset_config(dataset_name="train", config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, "dataset")
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(
                self._trainer.limit_train_batches, float
            ):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil(
                        (len(self._train_dl.dataset) / self.world_size)
                        / train_data_config["batch_size"]
                    )
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if "shuffle" not in val_data_config:
            val_data_config["shuffle"] = False

        # preserve config
        self._update_dataset_config(dataset_name="validation", config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if "shuffle" not in test_data_config:
            test_data_config["shuffle"] = False

        # preserve config
        self._update_dataset_config(dataset_name="test", config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, "_sample_rate"):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()

        return {
            "input_signal": NeuralType(("B", "T"), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple("B"), LengthsType(), optional=True),
            "processed_signal": NeuralType(
                ("B", "D", "T"), SpectrogramType(), optional=True
            ),
            "processed_signal_length": NeuralType(
                tuple("B"), LengthsType(), optional=True
            ),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(("B", "D", "T"), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple("B"), LengthsType()),
        }
