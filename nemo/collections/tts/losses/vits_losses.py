import torch
from torch.nn import functional as F

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LossType, VoidType
from nemo.core.neural_types.neural_type import NeuralType


class FeatureLoss(Loss):
    def input_types(self):
        return {
            "fmap_r": [[NeuralType(elements_type=VoidType())]],
            "fmap_g": [[NeuralType(elements_type=VoidType())]],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    # @typecheck()
    def forward(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2


class DiscriminatorLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_real_outputs": [NeuralType(('B', 'T'), VoidType())],
            "disc_generated_outputs": [NeuralType(('B', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
            "real_losses": [NeuralType(elements_type=LossType())],
            "fake_losses": [NeuralType(elements_type=LossType())],
        }

    def forward(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            dr = dr.float()
            dg = dg.float()
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses


class GeneratorLoss(Loss):
    """Generator Loss module"""

    @property
    def input_types(self):
        return {
            "disc_outputs": [NeuralType(('B', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
            "fake_losses": [NeuralType(elements_type=LossType())],
        }

    @typecheck()
    def forward(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses


class KlLoss(Loss):
    @property
    def input_types(self):
        return {
            "z_p": [NeuralType(('B', 'D', 'T'), VoidType())],
            "logs_q": [NeuralType(('B', 'D', 'T'), VoidType())],
            "m_p": [NeuralType(('B', 'D', 'T'), VoidType())],
            "logs_p": [NeuralType(('B', 'D', 'T'), VoidType())],
            "z_mask": [NeuralType(('B', 'D', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, z_p, logs_q, m_p, logs_p, z_mask):
        """
        z_p, logs_q: [b, h, t_t]
        m_p, logs_p: [b, h, t_t]
        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()

        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)
        return l
