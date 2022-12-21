import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
import torch
from typing import Union
from pytorch_lightning.trainer.trainer import Trainer
from collections import Counter
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as ptl
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from collections import Counter
from torch.utils.data import Dataset, DataLoader


from nemo.collections.nlp.models.language_modeling.megatron_gpt_adapter_model import MegatronGPTAdapterLearningModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner

def get_word_embedding(path):
    trainer = Trainer(strategy=NLPDDPStrategy(), devices=1, num_nodes=1, accelerator='gpu', precision=16, logger=False)
    save_restore_connector = NLPSaveRestoreConnector()
    frozen_model = MegatronGPTModel.restore_from(
                    path,
                    save_restore_connector=save_restore_connector,trainer=trainer,
                )
    word_embeddings = frozen_model.model.language_model.embedding.word_embeddings.weight.data
    tokenizer = frozen_model.tokenizer
    return word_embeddings, tokenizer


class EmbeddingProjector(ptl.LightningModule):
    def __init__(self, input_size:int , hidden_size:int, output_size:int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.GELU(), nn.Dropout(0.1))
        self.decoder = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(self.hidden_size, self.output_size))
        self.val_loss = float('inf')
        self.cs_loss = -1.0
        self.best_val_loss = float('inf')
        self.val_step = 0
        self.cs_embedding_loss = nn.CosineEmbeddingLoss(reduction='mean')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat, z

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def training_step(self, batch, batch_idx):
        x, y, y_neighbors = batch
        batch_size, inp_size = x.shape
        _, out_size = y.shape
        assert inp_size == self.input_size
        assert out_size == self.output_size
        y_hat, z = self(x)
        sl1_loss = F.smooth_l1_loss(y_hat, y)
        cs_loss = self.cs_embedding_loss(y_hat, y, torch.ones(batch_size))
        cs_neighbors_loss = self.cs_embedding_loss(y_hat, y_neighbors, -torch.ones(batch_size))
        total_loss = sl1_loss + cs_loss + cs_neighbors_loss
        self.log("sl1_loss", sl1_loss.item(), prog_bar=True)
        self.log("cs_loss", cs_loss.item(), prog_bar=True)
        self.log("train_loss", total_loss.item(), prog_bar=True)
        return total_loss

    def on_validation_epoch_start(self):
        self.val_step = 0

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, losses):
        sl1_losses = [i[0] for i in losses]
        cs_losses = [i[1] for i in losses]
        test_sl1_loss = sum(sl1_losses) / len(sl1_losses)
        test_cs_loss = sum(cs_losses) / len(cs_losses)
        test_loss = test_cs_loss + test_sl1_loss
        self.log("test_sl1_loss", test_sl1_loss, prog_bar=True, on_epoch=True)
        self.log("test_cs_loss", test_cs_loss, prog_bar=True, on_epoch=True)
        self.log("test_loss", test_loss, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y, y_neighbors = batch
        batch_size, inp_size = x.shape
        loss = None
        with torch.no_grad():
            y_hat, z = self(x)
            sl1_loss = F.smooth_l1_loss(y_hat, y)
            cs_loss = self.cs_embedding_loss(y_hat, y, torch.ones(batch_size))
            cs_neighbors_loss = self.cs_embedding_loss(y_hat, y_neighbors, -torch.ones(batch_size))
        return sl1_loss.item(), cs_loss.item(), cs_neighbors_loss.item()
    
    def validation_epoch_end(self, losses):
        sl1_losses = [i[0] for i in losses]
        cs_losses = [i[1] for i in losses]
        cs_neighbors_losses = [i[2] for i in losses]
        val_sl1_loss = sum(sl1_losses) / len(sl1_losses)
        val_cs_loss = sum(cs_losses) / len(cs_losses)
        val_cs_neighbors_loss = sum(cs_neighbors_losses) / len(cs_neighbors_losses)
        self.val_loss = val_sl1_loss + val_cs_loss + val_cs_neighbors_loss
        self.log("val_sl1_loss", val_sl1_loss, prog_bar=True, on_epoch=True)
        self.log("val_cs_loss", val_cs_loss, prog_bar=True, on_epoch=True)
        self.log("val_cs_neighbors_loss", val_cs_neighbors_loss, prog_bar=True, on_epoch=True)
        self.log("val_loss", self.val_loss, prog_bar=True, on_epoch=True)

        if self.val_loss < self.best_val_loss:
            print(f"new best loss found:{self.val_loss}")
            self.best_val_loss = self.val_loss
            self.save_model()
        self.log("best_val_loss", self.best_val_loss, prog_bar=True, on_epoch=True)
    
    def save_model(self):
        torch.save(self.state_dict(), 'model.pt')
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class UpscaleDataset(Dataset):
    def __init__(self, x_embeddings:torch.Tensor, y_embeddings:torch.Tensor) -> None:
        super().__init__()
        self.x_embs = x_embeddings
        self.y_embs = y_embeddings 
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
    
    
def load_prompt_learning_model(virtual_prompt_model_file):
    trainer = Trainer(strategy=NLPDDPStrategy(), devices=1, num_nodes=1, accelerator='gpu', precision=16, logger=False)
    prompt_learning_cfg = MegatronGPTPromptLearningModel.restore_from(
        virtual_prompt_model_file, trainer=trainer, return_config=True,
    )

    model = MegatronGPTPromptLearningModel.restore_from(
        restore_path=virtual_prompt_model_file, trainer=trainer, override_config_path=prompt_learning_cfg,
    )
    return model

def load_dataset(tokenizer, path):
    dataset = open(path, 'r', encoding='utf-8')
    for json_line in tqdm(dataset):
        doc = json.loads(json_line)
    

@hydra_runner(config_path="./", config_name="upscale")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    word_embeddings_125m, tokenizer = get_word_embedding(cfg.small_model_path)
    word_embeddings_1_3b, tokenizer = get_word_embedding(cfg.large_model_path)
    load_dataset(tokenizer, cfg.dataset)


    model_125m = load_prompt_learning_model(cfg.small_prompt_learning_model)
    prompt_learning_embs_125m = model_125m.prompt_table.prompt_table[cfg.taskname].prompt_embeddings.weight.data
    model_1_3b = load_prompt_learning_model(cfg.large_prompt_learning_model)
    
    train = UpscaleDataset(word_embeddings_125m[1000:2000], word_embeddings_1_3b[1000:2000])
    val = UpscaleDataset(word_embeddings_125m[:500], word_embeddings_1_3b[:500])
    test = UpscaleDataset(word_embeddings_125m[500:1000], word_embeddings_1_3b[500:1000])
    train_dataloader = DataLoader(train, batch_size=12, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=12, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=12, shuffle=False)
    projector = EmbeddingProjector(word_embeddings_125m.shape[1], 3000, word_embeddings_1_3b.shape[1])
    projector.cuda()
    trainer = ptl.Trainer(max_epochs=1, val_check_interval=0.3, num_sanity_val_steps=0)
    trainer.test(model=projector, dataloaders=test_dataloader)
    trainer.fit(model=projector, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    projector_2 = EmbeddingProjector(word_embeddings_125m.shape[1], 3000, word_embeddings_1_3b.shape[1])
    projector_2.load_model('model.pt')
    trainer.test(model=projector_2, dataloaders=test_dataloader)
    projector_2 = projector_2.cuda()

    y_hat, _ = projector_2(prompt_learning_embs_125m)
    print(y_hat.shape)

    model_1_3b.prompt_table.prompt_table[cfg.taskname].prompt_embeddings.weight.data = y_hat
    model_1_3b.save_to(cfg.projected_prompt_learning_path)



if __name__ == '__main__':
    main()
