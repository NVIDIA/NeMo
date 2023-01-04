import numpy as np
import os
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from apex.transformer import parallel_state
from omegaconf import OmegaConf
from typing import Any
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.core.config import hydra_runner
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


def load_prompt_learning_model(virtual_prompt_model_file, trainer_cfg):
    trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_cfg)
    prompt_learning_cfg = MegatronGPTPromptLearningModel.restore_from(
        virtual_prompt_model_file, trainer=trainer, return_config=True,
    )

    model = MegatronGPTPromptLearningModel.restore_from(
        restore_path=virtual_prompt_model_file, trainer=trainer, override_config_path=prompt_learning_cfg,
    )
    return model


def get_word_embedding(model):
    word_embeddings = model.frozen_model.model.language_model.embedding.word_embeddings.weight.data
    tokenizer = model.frozen_model.tokenizer
    return word_embeddings, tokenizer, model


def get_dataset(model, cfg):
    dataset = GPTPromptLearningDataset(
        data=[cfg.train_dataset],
        tokenizer=model.tokenizer,
        virtual_prompt_source=model.virtual_prompt_source,
        task_templates=model.task_templates,
        pseudo_tokens=model.pseudo_tokens,
        pad_token_id=model.pad_token_id,
        max_seq_length=1024,
        min_seq_length=1,
        add_bos=True,
        add_eos=True,
        for_train=True,
        tokens_to_generate=None,
        cache_data_path=None,
        load_cache=None,
    )
    _, tokens = zip(*sorted([(value, key) for (key, value) in dataset.counter.items()], reverse=True))
    tokens = tokens[cfg.high_freq_cutoff: cfg.num_examples + cfg.high_freq_cutoff]
    val_tokens = [i for idx, i in enumerate(tokens) if idx % 5 == 0]  # 20 % used for validation
    train_tokens = [i for idx, i in enumerate(tokens) if idx % 5 != 0]
    return train_tokens, val_tokens


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


class UpscaleDataset(Dataset):
    def __init__(self, precision:torch.dtype, x_embeddings: torch.Tensor, y_embeddings: torch.Tensor) -> None:
        super().__init__()
        self.x_embs = x_embeddings.type(precision)
        self.y_embs = y_embeddings.type(precision)
        assert self.x_embs.shape[0] == self.y_embs.shape[0]
        self.vocab = np.arange(0, self.y_embs.shape[0], dtype=int)
        x_embeddings_norm = x_embeddings / x_embeddings.norm(dim=1).unsqueeze(1)
        x_cs = x_embeddings_norm @ x_embeddings_norm.transpose(0, 1)
        self.x_sim_probs = torch.softmax(x_cs, dim=1)
        y_embeddings_norm = y_embeddings / y_embeddings.norm(dim=1).unsqueeze(1)
        y_cs = y_embeddings_norm @ y_embeddings_norm.transpose(0, 1)
        y_cs.fill_diagonal_(-float('inf'))
        self.y_sim_probs = torch.softmax(y_cs, dim=1).cpu().numpy()

    def __len__(self,):
        return self.x_embs.shape[0]

    def __getitem__(self, idx):
        k_neighbors = np.random.choice(self.vocab, 1, p=self.y_sim_probs[idx], replace=False)
        assert idx not in k_neighbors
        return self.x_embs[idx], self.y_embs[idx], self.y_embs[k_neighbors[0]]


def do_inference(model, trainer_cfg, inference_cfg, eval_dataset, projected_pred_file_path):
    trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_cfg)
    if parallel_state.is_unitialized():

        def placeholder():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(placeholder, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    # model_1_3b.save_to(cfg.projected_prompt_learning_path)
    length_params: LengthParam = {
        "max_length": inference_cfg.tokens_to_generate,
        "min_length": inference_cfg.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": inference_cfg.greedy,
        "temperature": inference_cfg.temperature,
        "top_k": inference_cfg.top_k,
        "top_p": inference_cfg.top_p,
        "repetition_penalty": inference_cfg.repetition_penalty,
        "add_BOS": inference_cfg.add_BOS,
        "all_probs": inference_cfg.all_probs,
        "compute_logprob": inference_cfg.compute_logprob,
    }

    max_input_length = model.frozen_model.cfg.encoder_seq_length - length_params["max_length"]
    _, dataloader = model.build_virtual_prompt_dataset(
        data=[eval_dataset],
        batch_size=inference_cfg.get("batch_size", 1),
        max_seq_length=max_input_length,
        min_seq_length=model.cfg.data.get('min_seq_length', 1),
        add_bos=sampling_params["add_BOS"],
        add_eos=False,
        for_train=False,
        tokens_to_generate=length_params["max_length"],
        drop_last=False,
        shuffle=False,
        zero_shot_baseline=False,
    )

    config = OmegaConf.to_container(inference_cfg)
    model.set_inference_config(config)
    response = trainer.predict(model, dataloader)
    projected_pred_file_path += ('.zs' if zero_shot_mode else '')
    with open(projected_pred_file_path, "w", encoding="utf-8") as pred_file:
        for i in range(len(response)):
            for sent in response[i]["sentences"]:
                sent = sent.strip()
                sent = sent.replace("\n", " ")
                pred_file.write(sent + "\n")
    return True


@hydra_runner(config_path="./", config_name="upscale")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    model_125m = load_prompt_learning_model(cfg.small_prompt_learning_model, cfg.nemo_trainer)
    word_embeddings_125m, tokenizer, model_125m = get_word_embedding(model_125m)
    prompt_learning_embs_125m = model_125m.prompt_table.prompt_table[cfg.taskname].prompt_embeddings.weight.data
    train_tokens, val_tokens = get_dataset(model_125m, cfg.upscaler.data)

    model_1_3b = load_prompt_learning_model(cfg.large_prompt_learning_model, cfg.nemo_trainer)
    word_embeddings_1_3b, tokenizer, model_1_3b = get_word_embedding(model_1_3b)
    prompt_learning_embs_1_3b = model_1_3b.prompt_table.prompt_table[cfg.taskname].prompt_embeddings.weight.data

    train = UpscaleDataset(torch.float16, word_embeddings_125m[train_tokens, :], word_embeddings_1_3b[train_tokens, :])
    val = UpscaleDataset(torch.float16, word_embeddings_125m[val_tokens, :], word_embeddings_1_3b[val_tokens, :])
    test = UpscaleDataset(torch.float16, prompt_learning_embs_125m, prompt_learning_embs_1_3b)

    train_dataloader = DataLoader(train, batch_size=cfg.upscaler.data.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=cfg.upscaler.data.batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=cfg.upscaler.data.batch_size, shuffle=False)

    projector = EmbeddingProjector(
        word_embeddings_125m.shape[1], word_embeddings_1_3b.shape[1], cfg.upscaler
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10, verbose=True, mode="min")
    wblogger = WandbLogger(**cfg.upscaler.wandb)
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=cfg.upscaler.save_checkpoint_path,
        filename="upscaler-{global_step}-{val_loss:.3f}_{val_cs_loss:.3f}_{val_csn_loss:.3f}_{val_sl1_loss:.4f}",
    )
    trainer = ptl.Trainer(**cfg.upscaler.trainer, callbacks=[early_stop_callback,checkpoint_callback], logger=wblogger)
    trainer.test(model=projector, dataloaders=test_dataloader)
    trainer.fit(model=projector, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_projector = EmbeddingProjector(
        word_embeddings_125m.shape[1], word_embeddings_1_3b.shape[1], cfg.upscaler
    )
    best_projector.load_model(cfg.upscaler.save_checkpoint_path + '/upscaler.pt')
    trainer.test(model=projector, dataloaders=test_dataloader)
    best_projector = best_projector.cuda()
    y_hat = best_projector(prompt_learning_embs_125m)

    model_1_3b.prompt_table.prompt_table[cfg.taskname].prompt_embeddings.weight.data = y_hat
    do_inference(model_1_3b, cfg.nemo_trainer, cfg.inference, cfg.upscaler.data.eval_dataset, cfg.projected_pred_file_path)
    model_1_3b.save_to(cfg.projected_prompt_learning_model)


if __name__ == '__main__':
    main()
