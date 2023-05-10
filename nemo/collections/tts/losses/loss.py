import torch

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LengthsType, LossType, NeuralType, PredictionsType, RegressionValuesType


class MaskedLoss(Loss):
    def __init__(self, loss_type: str, loss_scale: float = 1.0):
        super(MaskedLoss, self).__init__()
        self.loss_scale = loss_scale

        if loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss(reduction='none')
        elif loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type {loss_type}")

    @property
    def input_types(self):
        return {
            "target": NeuralType(('B', 'D', 'T'), RegressionValuesType()),
            "predicted": NeuralType(('B', 'D', 'T'), PredictionsType()),
            "target_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, predicted, target, target_len):
        assert target.shape[2] == predicted.shape[2]

        # [B, D, T]
        loss = self.loss_fn(input=predicted, target=target)
        # [B, T]
        loss = torch.mean(loss, dim=1)
        # [B]
        loss = torch.sum(loss, dim=1) / torch.clamp(target_len, min=1.0)

        # [1]
        loss = torch.mean(loss)
        loss = self.loss_scale * loss

        return loss
