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

from typing import List, Union, Optional

from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import PromptEncoderAdapterConfig
from nemo.collections.nlp.parts.peft_config import PEFTConfig
from nemo.core.classes.mixins.adapter_mixins import (
    AdapterModelPTMixin,
    AdapterModuleMixin,
    _prepare_default_adapter_config,
)
from nemo.utils import logging, model_utils


class NLPAdapterModelMixin(AdapterModelPTMixin):
    """ NLP Adapter Mixin that can augment any Encoder module with Adapter module support.
    # Todo rewrite doc string
    This mixin class should be used only with a top level ModelPT subclass, that includes an `encoder` submodule.
    This mixin class adds several utility methods which are propagated to the `encoder`.

    An Adapter module is any Pytorch nn.Module that possess a few properties :

    - It's input and output dimension are the same, while the hidden dimension need not be the same.
    - The final layer of the Adapter module is zero-initialized, so that the residual connection to the adapter
        yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `adapter_layer`: A torch.nn.ModuleDict(), whose keys are the names of the adapter (globally unique),
                and values are the Adapter nn.Module().
        -   `adapter_cfg`: A OmegaConf DictConfig object that holds the config of the adapters that are initialized.
        -   `adapter_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
                The value resolves to `global_cfg`, and can be overridden via `model.cfg.adapters.global_cfg.*`.

    **Note**: This module **is** responsible for maintaining its config. At the ModelPT level, it will access and
        write Adapter config information to `self.cfg.adapters`.
    """

    def __init__(self, *args, **kwargs):
        self.use_peft = False
        self.setup_complete = False
        super().__init__(*args, **kwargs)
        if hasattr(self, "enc_dec_model"):
            self.model_prefix = "enc_dec_model."  # for T5
        else:
            self.model_prefix = "model.module." if self.cfg.megatron_amp_O2 else "model."

    @classmethod
    def restore_cfg(
        cls,
        restore_path: str,
        overwrites: Optional[OmegaConf] = None,
    ) -> OmegaConf:
        output = cls.restore_from(restore_path, return_config=True)

        if overwrites:
            OmegaConf.resolve(overwrites)
            with open_dict(output):
                for key, val in overwrites.model.items():
                    output[key] = val
                if "test_ds" in overwrites.model.data:
                    output.micro_batch_size = overwrites.model.data.train_ds.micro_batch_size
                    output.global_batch_size = overwrites.model.data.train_ds.global_batch_size
                if overwrites.get("trainer", None) and overwrites.trainer.get("precision"):
                    output.precision = overwrites.trainer.precision

        return output

    @classmethod
    def load(
        cls,
        to_load: Union["str", OmegaConf],
        trainer: Optional[Trainer] = None,
        peft: Optional[Union[OmegaConf, PEFTConfig]] = None
    ):
        if isinstance(to_load, str):
            return cls.restore_from(to_load, trainer=trainer)

        restore_path = to_load.model.restore_from_path
        override_cfg = cls.restore_cfg(restore_path, to_load)

        output = cls.restore_from(
            restore_path, override_cfg, trainer=trainer
        )

        if peft:
            AdapterConfig = PEFT_CONFIG_MAP[peft.peft_scheme]
            output.add_adapter(AdapterConfig(override_cfg))

        return output

    def _get_all_keys(self,):
        """
        Returns all the keys in the model
        """
        k = [n for n, p in self.named_parameters()]
        return set(k)

    def add_adapter(self, peft_cfgs: Union[PEFTConfig, List[PEFTConfig]]):
        """
        High level API to add one or more adapter modules to the model, and freeze the base weights

        Args:
            peft_cfgs: One or more PEFTConfig objects that specify the PEFT method configuration
        """

        def _check_and_add_adapter(module, peft_name, peft_cfg):
            if isinstance(module, AdapterModuleMixin):
                if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
                    module.add_adapter(name=peft_name, cfg=peft_cfg)

        if not isinstance(peft_cfgs, List):
            peft_cfgs = [peft_cfgs]

        self.base_keys = self._get_all_keys()
        self.freeze()
        logging.info(f"Before adding PEFT params:\n{self.summarize()}")

        for cfg in peft_cfgs:
            layer_selection = cfg.layer_selection
            for peft_name, peft_cfg in cfg.get_config_dict().items():
                # self.model.language_model means is GPT and not T5
                if (
                    hasattr(self, "model")
                    and hasattr(self.model, "language_model")
                    and not isinstance(peft_cfg, PromptEncoderAdapterConfig)
                ):
                    if layer_selection is not None:
                        logging.info(
                            f"Layer selection {layer_selection} is enabled for the current model ("
                            f"{self.__class__.__name__} + {peft_name})"
                        )
                    for layer in self.model.language_model.encoder.layers:
                        if layer.layer_number in layer_selection:
                            for _, module in layer.named_modules():
                                _check_and_add_adapter(module, peft_name, peft_cfg)
                else:
                    # Non GPT models, as well as GPT+PTuning do not support layer selection
                    if layer_selection is not None:
                        logging.warning(
                            "Layer selection is specified, but it is not supported for either "
                            f"{self.__class__.__name__} or {peft_name})"
                        )
                    for _, module in self.named_modules():
                        _check_and_add_adapter(module, peft_name, peft_cfg)

                # Update the model.cfg with information about the new adapter from cfg
                module_name, adapter_name = self.resolve_adapter_module_name_(peft_name)
                with open_dict(self.cfg):
                    # Construct the minimum config required to be updated by adapter implementations
                    if 'adapters' not in self.cfg:
                        self.cfg.adapters = OmegaConf.create({})

                    self.cfg.adapters = _prepare_default_adapter_config(
                        global_key=self.adapter_global_cfg_key,
                        meta_key=self.adapter_metadata_cfg_key,
                        cfg=self.cfg.adapters,
                    )

                    # Inject the module name in the adapter metadata cfg
                    gcfg = self.adapter_global_cfg_key
                    mcfg = self.adapter_metadata_cfg_key
                    self.cfg.adapters[gcfg][mcfg]['modules'][adapter_name] = module_name

                    self.cfg.adapters[adapter_name] = OmegaConf.create(peft_cfg)

        logging.info(f"After adding PEFT params:\n{self.summarize()}")
        self.adapter_keys = self._get_all_keys() - self.base_keys

        for cfg in peft_cfgs:
            if cfg.weight_tying:
                self.tie_weights(cfg)
        self.use_peft = True

    def tie_weights(self, peft_cfg):
        pos_idx = 0

        if isinstance(peft_cfg, LoraPEFTConfig):
            layer0 = self.model.language_model.encoder.layers[0].self_attention
        elif isinstance(peft_cfg, AdapterPEFTConfig):
            layer0 = self.model.language_model.encoder.layers[0]
        else:
            raise RuntimeError(f"{peft_cfg} is not supported for tied weights")

        for adapter_name in layer0.adapter_layer:
            adapter = layer0.get_adapter_module(adapter_name)
            print(adapter_name, pos_idx)
            adapter.set_position(pos_idx)
            pos_idx += 1

        for layer in self.model.language_model.encoder.layers[1:]:
            if isinstance(peft_cfg, LoraPEFTConfig):
                layer = layer.self_attention
            for adapter_name in layer.adapter_layer:
                print(adapter_name, pos_idx)
                adapter_l = layer.get_adapter_module(adapter_name)
                adapter_0 = layer0.get_adapter_module(adapter_name)
                adapter_l.tie_weights(pos_idx, adapter_0)
                pos_idx += 1

    def get_peft_state_dict(self):
        """
        Gets the keys associated with the adapters only.
        """
        state_dict = self.model.state_dict(prefix=self.model_prefix)
        peft_state_dict = {}
        for k in self.adapter_keys:
            # state_dict keys needs to be in non-O2 format and will be corrected in PEFTSaveRestoreConnector if O2=True
            new_k = k.replace("model.module.", "model.", 1)
            peft_state_dict[new_k] = state_dict[k]
        return peft_state_dict


    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.use_peft and self.setup_complete:
            # Once setup is complete we no longer need to track the frozen part of the model. Only there adapter state dict keeps changing so state_dict only track these.
            return self.get_peft_state_dict()
        else:
            # we want all the params with the same keys as calling self.state_dict()
            # but we can't call self.state_dict() here as it would be a recursive call.
            # so we call self.model.state_dict(prefix="model.") which will return all the keys and params same as calling self.state_dict()
            return self.model.state_dict(prefix=self.model_prefix)

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.use_peft and self.setup_complete:
            # at this stage only adapter params will appear in the state_dict arg
            # so we only update those while the rest of the model is frozen.
            # setting strict=False will ignore the missing keys (which are not being updated anyway)
            # explicitly check if state_dict.keys matches all the expected self.adapter_keys since we don't have the
            # safety in strict=True anymore.
            assert set(state_dict.keys()) == self.adapter_keys
            super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict, strict=True)
