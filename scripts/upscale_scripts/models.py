import os
from typing import Any

import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from torch import nn


class EmbeddingLinearCombination(ptl.LightningModule):
    def __init__(self, num_word_tokens: int, embed_dim: int, num_virtual_prompts: int, cfg):
        super().__init__()
        self.num_word_tokens = num_word_tokens
        self.num_prompt_tokens = num_virtual_prompts
        t = torch.randn((self.num_word_tokens, self.num_prompt_tokens))
        self.linear_combination = torch.nn.parameter.Parameter(data=t)
        self.cs_embedding_loss = nn.CosineEmbeddingLoss(reduction='mean')
        self.cs_wt = cfg.get("cs_loss_weight", 1.0)
        self.sl1_wt = cfg.get("sl1_loss_weight", 1.0)
        self.l2_reg_wt = cfg.get("l2_loss_weight", 1.0)
        self.l1_reg_wt = cfg.get("l1_loss_weight", 1.0)
        self.save_pt_path = cfg.save_checkpoint_path + '/embedding_linear_combination.pt'
        self.val_loss = float('inf')
        self.cs_loss = -1.0
        self.best_val_loss = float('inf')
        self.val_step = 0

    def is_cuda(self,):
        return all(p.is_cuda for p in self.parameters())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def forward(self, token_embeds) -> Any:
        w = self.linear_combination
        output_embeds = (token_embeds.transpose(0, 1) @ w).transpose(0, 1)
        l2 = (w ** 2).mean()
        l1 = torch.abs(w).mean()
        return output_embeds, l2, l1

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze()
        y = y.squeeze()
        y_hat, l2_reg, l1_reg = self(x)
        sl1_loss = F.smooth_l1_loss(y_hat, y)
        cs_loss = self.cs_embedding_loss(y_hat, y, torch.ones(y.shape[0]).type_as(y_hat))

        total_loss = (
            (self.sl1_wt * sl1_loss) + (self.cs_wt * cs_loss) + (self.l1_reg_wt * l1_reg) + (self.l2_reg_wt * l2_reg)
        )

        self.log("sl1_loss", sl1_loss.item(), prog_bar=True)
        self.log("cs_loss", cs_loss.item(), prog_bar=True)
        self.log("l1_reg", l1_reg.item(), prog_bar=True)
        self.log("l2_reg", l2_reg.item(), prog_bar=True)
        self.log("loss", total_loss.item(), prog_bar=True)
        self.log("global_step", self.global_step, prog_bar=True)
        return total_loss

    def on_validation_epoch_start(self):
        self.val_step = 0

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, losses):
        sl1_losses = [i[0] for i in losses]
        cs_losses = [i[1] for i in losses]
        l2_regs = [i[2] for i in losses]
        l1_regs = [i[3] for i in losses]
        sl1_loss = sum(sl1_losses) / len(sl1_losses)
        cs_loss = sum(cs_losses) / len(cs_losses)
        l1_reg = sum(l1_regs) / len(l1_regs)
        l2_reg = sum(l2_regs) / len(l2_regs)
        test_loss = (
            (self.sl1_wt * sl1_loss) + (self.cs_wt * cs_loss) + (self.l1_reg_wt * l1_reg) + (self.l2_reg_wt * l2_reg)
        )
        self.log("test_sl1_loss", sl1_loss, prog_bar=True, on_epoch=True)
        self.log("test_cs_loss", cs_loss, prog_bar=True, on_epoch=True)
        self.log("test_loss", test_loss, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze()
        y = y.squeeze()
        batch_size, inp_size = y.shape
        loss = None
        with torch.no_grad():
            y_hat, l2_reg, l1_reg = self(x)
            sl1_loss = F.smooth_l1_loss(y_hat, y)
            cs_loss = self.cs_embedding_loss(y_hat, y, torch.ones(y.shape[0]).type_as(y_hat))
        return sl1_loss.item(), cs_loss.item(), l2_reg.item(), l1_reg.item()

    def validation_epoch_end(self, losses):
        sl1_losses = [i[0] for i in losses]
        cs_losses = [i[1] for i in losses]
        l2_regs = [i[2] for i in losses]
        l1_regs = [i[3] for i in losses]
        sl1_loss = sum(sl1_losses) / len(sl1_losses)
        cs_loss = sum(cs_losses) / len(cs_losses)
        l1_reg = sum(l1_regs) / len(l1_regs)
        l2_reg = sum(l2_regs) / len(l2_regs)
        self.val_loss = (
            (self.sl1_wt * sl1_loss) + (self.cs_wt * cs_loss) + (self.l1_reg_wt * l1_reg) + (self.l2_reg_wt * l2_reg)
        )
        self.log("val_sl1_loss", sl1_loss, prog_bar=True, on_epoch=True)
        self.log("val_cs_loss", cs_loss, prog_bar=True, on_epoch=True)
        self.log("val_loss", self.val_loss, prog_bar=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True)
        if self.val_loss < self.best_val_loss:
            self.save_model()

    def save_model(self):
        dir = os.path.dirname(self.save_pt_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.state_dict(), self.save_pt_path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class EmbeddingProjector(ptl.LightningModule):
    def __init__(self, input_size: int, output_size: int, cfg: Any) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = cfg.hidden_size
        self.upscaler = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.output_size),
        )
        self.val_loss = float('inf')
        self.cs_loss = -1.0
        self.best_val_loss = float('inf')
        self.val_step = 0
        self.cs_embedding_loss = nn.CosineEmbeddingLoss(reduction='mean')
        self.cs_wt = cfg.get("cs_loss_weight", 1.0)
        self.csn_wt = cfg.get("csn_loss_weight", 1.0)
        self.sl1_wt = cfg.get("sl1_loss_weight", 1.0)
        self.save_pt_path = cfg.save_checkpoint_path + '/upscaler.pt'

    def is_cuda(self,):
        return all(p.is_cuda for p in self.parameters())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def forward(self, x):
        y_hat = self.upscaler(x)
        return y_hat

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def training_step(self, batch, batch_idx):
        x, y, y_neighbors = batch
        batch_size, inp_size = x.shape
        _, out_size = y.shape
        assert inp_size == self.input_size
        assert out_size == self.output_size
        y_hat = self(x)
        sl1_loss = F.smooth_l1_loss(y_hat, y)
        cs_loss = self.cs_embedding_loss(y_hat, y, torch.ones(batch_size).type_as(y_hat))
        cs_neighbors_loss = self.cs_embedding_loss(y_hat, y_neighbors, -torch.ones(batch_size).type_as(y_hat))

        total_loss = (self.sl1_wt * sl1_loss) + (self.cs_wt * cs_loss) + (self.csn_wt * cs_neighbors_loss)

        self.log("sl1_loss", sl1_loss.item(), prog_bar=True)
        self.log("cs_loss", cs_loss.item(), prog_bar=True)
        self.log("csn_loss", cs_neighbors_loss.item(), prog_bar=True)
        self.log("loss", total_loss.item(), prog_bar=True)
        self.log("global_step", self.global_step, prog_bar=True)
        return total_loss

    def on_validation_epoch_start(self):
        self.val_step = 0

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, losses):
        sl1_losses = [i[0] for i in losses]
        cs_losses = [i[1] for i in losses]
        csn_losses = [i[2] for i in losses]
        sl1_loss = sum(sl1_losses) / len(sl1_losses)
        cs_loss = sum(cs_losses) / len(cs_losses)
        csn_loss = sum(csn_losses) / len(csn_losses)
        total_loss = (self.sl1_wt * sl1_loss) + (self.cs_wt * cs_loss) + (self.csn_wt * csn_loss)
        self.log("test_sl1_loss", sl1_loss, prog_bar=True, on_epoch=True)
        self.log("test_cs_loss", cs_loss, prog_bar=True, on_epoch=True)
        self.log("test_csn_loss", csn_loss, prog_bar=True, on_epoch=True)
        self.log("test_loss", total_loss, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y, y_neighbors = batch
        batch_size, inp_size = x.shape
        loss = None
        with torch.no_grad():
            y_hat = self(x)
            sl1_loss = F.smooth_l1_loss(y_hat, y)
            cs_loss = self.cs_embedding_loss(y_hat, y, torch.ones(batch_size).type_as(y_hat))
            cs_neighbors_loss = self.cs_embedding_loss(y_hat, y_neighbors, -torch.ones(batch_size).type_as(y_hat))
        return sl1_loss.item(), cs_loss.item(), cs_neighbors_loss.item()

    def validation_epoch_end(self, losses):
        sl1_losses = [i[0] for i in losses]
        cs_losses = [i[1] for i in losses]
        csn_losses = [i[2] for i in losses]
        sl1_loss = sum(sl1_losses) / len(sl1_losses)
        cs_loss = sum(cs_losses) / len(cs_losses)
        csn_loss = sum(csn_losses) / len(csn_losses)
        self.val_loss = (self.sl1_wt * sl1_loss) + (self.cs_wt * cs_loss) + (self.csn_wt * csn_loss)
        self.log("val_sl1_loss", sl1_loss, prog_bar=True, on_epoch=True)
        self.log("val_cs_loss", cs_loss, prog_bar=True, on_epoch=True)
        self.log("val_csn_loss", csn_loss, prog_bar=True, on_epoch=True)
        self.log("val_loss", self.val_loss, prog_bar=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True)
        if self.val_loss < self.best_val_loss:
            self.save_model()

    def save_model(self):
        dir = os.path.dirname(self.save_pt_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.state_dict(), self.save_pt_path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
