import torch
import torch.nn as nn

class VitsGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_loss_alpha = 45
        self.gen_loss_alpha = 1
        self.feat_loss_alpha = 1
        self.dur_loss_alpha = 1
        self.mel_loss_alpha = 1

    @staticmethod
    def feature_loss(feats_real, feats_generated):
        loss = 0
        for dr, dg in zip(feats_real, feats_generated):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2

    @staticmethod
    def generator_loss(scores_fake):
        loss = 0
        gen_losses = []
        for dg in scores_fake:
            dg = dg.float()
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    @staticmethod
    def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
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
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)
        return l


    def forward(
        self,
        mel_slice,
        mel_slice_hat,
        z_p,
        logs_q,
        m_p,
        logs_p,
        z_mask,
        scores_disc_fake,
        feats_disc_fake,
        feats_disc_real,
        loss_duration,
    ):
        """
        Shapes:
            - mel_slice : :math:`[B, 1, T]`
            - mel_slice_hat: :math:`[B, 1, T]`
            - z_p: :math:`[B, C, T]`
            - logs_q: :math:`[B, C, T]`
            - m_p: :math:`[B, C, T]`
            - logs_p: :math:`[B, C, T]`
            - z_len: :math:`[B]`
            - scores_disc_fake[i]: :math:`[B, C]`
            - feats_disc_fake[i][j]: :math:`[B, C, T', P]`
            - feats_disc_real[i][j]: :math:`[B, C, T', P]`
        """
        loss = 0.0
        return_dict = {}
        # compute losses
        loss_kl = (
            self.kl_loss(z_p=z_p, logs_q=logs_q, m_p=m_p, logs_p=logs_p, z_mask=z_mask.unsqueeze(1))
            * self.kl_loss_alpha
        )
        loss_feat = (
            self.feature_loss(feats_real=feats_disc_real, feats_generated=feats_disc_fake) * self.feat_loss_alpha
        )
        loss_gen = self.generator_loss(scores_fake=scores_disc_fake)[0] * self.gen_loss_alpha
        loss_mel = torch.nn.functional.l1_loss(mel_slice, mel_slice_hat) * self.mel_loss_alpha
        loss_duration = torch.sum(loss_duration.float()) * self.dur_loss_alpha
        loss = loss_kl + loss_feat + loss_mel + loss_gen + loss_duration

        # pass losses to the dict
        return_dict["loss_gen"] = loss_gen
        return_dict["loss_kl"] = loss_kl
        return_dict["loss_feat"] = loss_feat
        return_dict["loss_mel"] = loss_mel
        return_dict["loss_duration"] = loss_duration
        return_dict["loss"] = loss
        return return_dict


class VitsDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def discriminator_loss(scores_real, scores_fake):
        loss = 0
        real_losses = []
        fake_losses = []
        for dr, dg in zip(scores_real, scores_fake):
            dr = dr.float()
            dg = dg.float()
            real_loss = torch.mean((1 - dr) ** 2)
            fake_loss = torch.mean(dg**2)
            loss += real_loss + fake_loss
            real_losses.append(real_loss.item())
            fake_losses.append(fake_loss.item())
        return loss, real_losses, fake_losses

    def forward(self, scores_disc_real, scores_disc_fake):
        loss = 0.0
        loss_disc, loss_disc_real, loss_disc_fake = self.discriminator_loss(
            scores_real=scores_disc_real, scores_fake=scores_disc_fake
        )

        return loss_disc, loss_disc_real, loss_disc_fake