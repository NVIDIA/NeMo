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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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

import os
import sys
import warnings
from typing import Optional, List, Dict

import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.core.config import hydra_runner
from nemo.core.classes import Exportable
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.utils import logging

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext

class MegatronGPTExportableModel(torch.nn.Module, Exportable):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dtype = None
        if model.cfg['precision'] == 'bf16':
            self.dtype = torch.bfloat16
        elif int(model.cfg['precision']) == 32:
            self.dtype = torch.float
        elif int(model.cfg['precision']) == 16:
            self.dtype = torch.float16
        else:
            raise ValueError(f"precision: {model.cfg['precision']} is not supported.")

    def forward(self, id_tensors, masks_and_position_ids):
        output_tensors = []
        for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
            attn_mask, _, pos_ids = attn_mask_and_pos_ids
            assert tokens.shape == pos_ids.shape
            assert attn_mask.shape[2] == attn_mask.shape[3] == tokens.shape[1] == pos_ids.shape[1]
            with torch.autocast('cuda', dtype=self.dtype):
                output_tensor = self.model.forward(
                        tokens=tokens.cuda(),
                        text_position_ids=pos_ids.cuda(),
                        attention_mask=attn_mask.cuda(),
                        labels=None,
                    )

            output_tensors.append(output_tensor)
        return output_tensors

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def input_example(self, max_batch=1, max_dim=768, seq_len=6):
        ids = [self.model.tokenizer.text_to_ids(text) for text in ['hi there']]
        id_tensors = [torch.unsqueeze(torch.LongTensor(id_list), dim=0) for id_list in ids]
        masks_and_position_ids = [
            get_ltor_masks_and_position_ids(id_tensor, self.model.tokenizer.eos_id, False, False, False)
            for id_tensor in id_tensors
        ]

        return id_tensors, masks_and_position_ids
    
    def get_dynamic_axes(self):
        dynamic_axes = {
                'id_tensors': {0: "BS", 1: "sequence"},
                'masks_and_position_ids': {0: "BS", 2: "sequence", 3: "sequence"},
        }
        return dynamic_axes
    
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "id_tensors": NeuralType(('B'), ChannelType()),
            "masks_and_position_ids": NeuralType(('B'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"log_probs": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ['id_tensors', 'masks_and_position_ids']

    @property
    def output_names(self) -> List[str]:
        return ['log_probs']

@hydra_runner(config_path="conf", config_name="megatron_gpt_export")
def nemo_export(cfg):
    """Convert a .nemo saved model into .onnx ONNX format."""
    nemo_in = cfg.gpt_model_file
    onnx_out = cfg.onnx_model_file

    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"


    logging.info("Restoring NeMo model from '{}'".format(nemo_in))
    try:
        with torch.inference_mode():
            # Restore instance from .nemo file using generic model restore_from
            save_restore_connector = NLPSaveRestoreConnector()
            if os.path.isdir(cfg.gpt_model_file):
                save_restore_connector.model_extracted_dir = cfg.gpt_model_file

            pretrained_cfg = MegatronGPTModel.restore_from(
                restore_path=cfg.gpt_model_file,
                trainer=trainer,
                return_config=True,
                save_restore_connector=save_restore_connector,
            )
            OmegaConf.set_struct(pretrained_cfg, True)
            with open_dict(pretrained_cfg):
                pretrained_cfg.sequence_parallel = False
                pretrained_cfg.activations_checkpoint_granularity = None
                pretrained_cfg.activations_checkpoint_method = None
                pretrained_cfg.precision = trainer.precision
            model = MegatronGPTModel.restore_from(
                restore_path=cfg.gpt_model_file,
                trainer=trainer,
                override_config_path=pretrained_cfg,
                save_restore_connector=save_restore_connector,
            )
    except Exception as e:
        logging.error(
            "Failed to restore model from NeMo file : {}. Please make sure you have the latest NeMo package installed with [all] dependencies.".format(
                nemo_in
            )
        )
        raise e

    logging.info("Model {} restored from '{}'".format(model.__class__.__name__, nemo_in))

    if not isinstance(model, Exportable):
        logging.error("Your NeMo model class ({}) is not Exportable.".format(model.__class__.__name__))
        sys.exit(1)

    #
    #  Add custom export parameters here
    #
    check_trace = cfg.export_options.runtime_check

    if cfg.export_options.cache_support and hasattr(model, "encoder") and hasattr(model.encoder, "export_cache_support"):
        model.encoder.export_cache_support = True
        logging.info("Caching support is enabled.")
        model.encoder.setup_streaming_params()

    autocast = nullcontext
    if cfg.export_options.autocast:
        autocast = torch.cuda.amp.autocast
    fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
    try:
        with autocast(), torch.no_grad(), torch.inference_mode(), te.onnx_export(True), \
            te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe), warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                category=torch.jit.TracerWarning,
                module=r'.*'
            )

            model.to(device=cfg.export_options.device).freeze()
            model.eval()
            exportable_model = MegatronGPTExportableModel(model)

            exportable_model.export(
                onnx_out,
                onnx_opset_version=cfg.export_options.onnx_opset,
                do_constant_folding=cfg.export_options.do_constant_folding,
                dynamic_axes=exportable_model.get_dynamic_axes(),
                check_trace=check_trace,
                check_tolerance=cfg.export_options.check_tolerance,
                verbose=cfg.export_options.verbose,
            )
    except Exception as e:
        logging.error(
            "Export failed. Please make sure your NeMo model class ({}) has working export() and that you have the latest NeMo package installed with [all] dependencies.".format(
                model.__class__
            )
        )
        raise e


if __name__ == '__main__':
    nemo_export()
