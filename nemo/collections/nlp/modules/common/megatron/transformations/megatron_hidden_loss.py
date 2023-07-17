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


import math
import torch

__all__ = ["MegatronBaseHiddenLoss", "MegatronAMIMHiddenLoss", "MegatronVAEHiddenLoss"]


class MegatronBaseHiddenLoss(torch.nn.Module):
    """
    Base class to calculate hidden state loss.
    Returned dict includes a loss value and additional outputs.
    """

    def __init__(self, loss_weight=1.0, name=""):
        super().__init__()
        self.name = name
        self.loss_weight = float(loss_weight)

    def __str__(self):
        return super().__str__() + f"(name={self.name})"

    def _validate_inputs(self, inputs):
        """Validate inputs"""
        # validate inputs
        if not set(self.input_names).isssubset(set(inputs.keys())):
            raise ValueError(f"Inputs should contain {self.input_names}, but got {inputs.keys()}")

    @property
    def input_names(self):
        """Returns and caches input names"""
        # we always expect hiddens_mask to be used to mask out loss of padded elements
        return self._input_name() + ["hiddens_mask"]

    def _input_names(self):
        """Add here all required inputs"""
        return []

    def _loss(self, inputs, batch_data=None):
        """
        Implement your own loss calculations. Must return "loss" key.
        loss shape - [B, S] for Batch, Sequence sizes
        batch_data - a dictionary of additional data that can be used to calculate loss
        """
        raise NotImplementedError("Please implement loss calculations in child class")
        # return {"loss": 0.0}

    def loss(self, inputs, batch_data=None):
        """A wrapper around custom _loss that adds a weighted loss and name to the output dict"""
        self._validate_inputs(inputs)

        loss_dict = self._loss(inputs, batch_data=batch_data)
        if "loss" not in loss_dict:
            raise KeyError("Loss dict must contain 'loss' key")

        # average loss over active steps only. loss [B, S]
        loss = loss_dict["loss"]
        hidden_mask = inputs["hidden_mask"].to(loss)
        loss = loss * hidden_mask
        # loss [B, S] -> [B] sequence level loss
        loss = loss.sum(dim=-1) / inputs["hidden_mask"].sum(dim=-1).clamp(min=1.0)

        # compute batch level weighted loss (scalar)
        weighted_loss = loss.sum() * self.loss_weight

        # store updated losses
        loss_dict["loss"] = loss
        loss_dict["weighted_loss"] = weighted_loss

        return loss_dict


class MegatronAMIMHiddenLoss(MegatronBaseHiddenLoss):
    """
    Based on <https://arxiv.org/abs/2003.02645>
    Implements A-MIM loss with a unit Normal anchor.
    A-MIM - asymmetric MIM (without sampling)
    """

    def __init__(self, loss_weight=1.0, hidden_aggregation_method="sum", name="mim"):
        super().__init__(
            name=name, loss_weight=loss_weight,
        )

        # allows to determine how to aggregate hidden loss over hidden dimension
        self.hidden_aggregation_method = hidden_aggregation_method

    def _input_names(self):
        """Add here all required inputs"""
        return ["z", "z_log_prob"]

    def _loss(self, inputs, batch_data=None):
        """
        Implement your own loss calculations. Must return "loss" key.
        loss shape - [B, S] for Batch, Sequence sizes
        """
        z = self.get_input(inputs, "z")
        # get posterior
        log_prob_q_z_given_x = self.get_input(inputs, "z_log_prob")
        # compute log prob of anchor a unit Normal distribution
        log_prob_P_z = -0.5 * (math.log(2 * math.pi) + z.pow(2))
        # aggregate over hidden dimension, default is sum
        log_prob_P_z = getattr(log_prob_P_z, self.hidden_aggregation_method)(dim=-1)

        # A-MIM loss = log_p_x_given_z - 0.5 * (log_prob_P_z + log_prob_q_z_given_x)
        # here we return only the hidden loss part
        loss = -0.5 * (log_prob_P_z + log_prob_q_z_given_x)

        return {
            "loss": loss,
            "log_prob_P_z": log_prob_P_z,
            "log_prob_q_z_given_x": log_prob_q_z_given_x,
        }


class MegatronVAEHiddenLoss(MegatronBaseHiddenLoss):
    """
    Based on <https://arxiv.org/abs/1312.6114>
    Implements VAE loss with a unit Normal anchor.
    """

    def __init__(self, loss_weight=1.0, min_kl_value=None, name="vae"):
        super().__init__(
            name=name, loss_weight=loss_weight,
        )

        # minimum value for KL divergence
        if min_kl_value is None:
            self.min_kl_value = min_kl_value
        else:
            self.min_kl_value = float(min_kl_value)

    def _input_names(self):
        """Add here all required inputs"""
        return ["z", "z_log_prob"]

    def _loss(self, inputs, batch_data=None):
        z = self.get_input(inputs, "z")
        # get posterior
        log_prob_q_z_given_x = self.get_input(inputs, "z_log_prob")
        # compute log prob of anchor a unit Normal distribution
        log_prob_p_z = -0.5 * (math.log(2 * math.pi) + z.pow(2)).sum(dim=-1)

        # VAE loss = log_p_x_given_z - KL(q(z|x) || p(z))
        kl_div = log_prob_q_z_given_x - log_prob_p_z
        # here we return only the hidden loss part
        loss = -kl_div

        return {
            "loss": loss,
            "kl_div": kl_div,
            "log_prob_p_z": log_prob_p_z,
            "log_prob_q_z_given_x": log_prob_q_z_given_x,
        }
