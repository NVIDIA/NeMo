# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import transformer_engine.pytorch as te

from nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm import HfAutoModelForCausalLM


class TEAccelerator:

    @staticmethod
    def accelerate(model: HfAutoModelForCausalLM):
        assert isinstance(model, HfAutoModelForCausalLM), "Only HFAutoModelForCauselLM class is supported."
        assert model.model_name is not None, "There is no Hugging Face model available."

        TEAccelerator._accelerate(model.hf_model)
        return model

    @staticmethod
    def _accelerate(model):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):

                # print(name, module)
                # print(module.weight)
                # print("")
                # print("")

                has_bias = module.bias is not None
                if any(p % 16 != 0 for p in module.weight.shape):
                    print("continuing")
                    continue
                te_module = te.Linear(
                    module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
                )
                with torch.no_grad():
                    te_module.weight.copy_(module.weight)
                    if has_bias:
                        te_module.bias.copy_(module.bias)

                setattr(model, name, te_module)
                # print(te_module.weight)
            TEAccelerator._accelerate(module)

        return model

    @staticmethod
    def te_accelerated(model: HfAutoModelForCausalLM):
        assert isinstance(model, HfAutoModelForCausalLM), "Only HFAutoModelForCauselLM class is supported."
        assert model.hf_model is not None, "There is no Hugging Face model available."
        return TEAccelerator._te_accelerated(model.hf_model)

    @staticmethod
    def _te_accelerated(model):
        for name, module in model.named_children():
            if isinstance(module, (te.LayerNorm, te.Linear, te.TransformerLayer)):
                return True
            else:
                if TEAccelerator._te_accelerated(module):
                    return True

        return False
