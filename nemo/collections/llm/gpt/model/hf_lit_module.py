import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from transformers import AutoModelForCausalLM

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning import io


def _extract_non_bias_params(model):
    return list(map(lambda x: x[1], filter(lambda x: not 'bias' in x[0], model.named_parameters())))


def masked_cross_entropy(logits, targets, mask=None):
    if mask is not None:
        loss = F.cross_entropy(logits, targets, reduction='none')
        return torch.mean(loss[mask == 1])
    else:
        return F.cross_entropy(logits, targets)


class HfAutoModel(pl.LightningModule, io.IOMixin):
    def __init__(
        self, model_name='gpt2', tokenizer=None, optim_cls=Adam, optim_conf={'lr': 1e-5}, loss_fn=masked_cross_entropy
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer(model_name)
        self.optim_conf = optim_conf
        self.optim_cls = optim_cls
        self.model = None
        self.loss_fn = loss_fn

    def configure_model(self):
        # create all your layers here
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype='auto')
        self.model.train()

    def forward(self, input_ids, attention_mask=None, labels=None, loss_mask=None):
        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask,
        )
        labels = labels.to(self.model.device)
        loss_mask = loss_mask.to(self.model.device)
        n_cls = outputs.logits.shape[-1]
        outputs.loss = self.loss_fn(outputs.logits.view(-1, n_cls), labels.view(-1), loss_mask.view(-1))
        return outputs

    def training_step(self, *args, **kwargs):
        tokens = args[0]['tokens']
        labels = args[0]['labels']
        loss_mask = args[0]['loss_mask']
        output = self.forward(
            input_ids=tokens,
            labels=labels,
            loss_mask=loss_mask,
        )

        loss = output.loss
        self.log('train_log', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens = args[0]['tokens']
        labels = args[0]['labels']
        output = self.forward(
            input_ids=tokens,
            labels=labels,
        )

        loss = output.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        assert self.model is not None, "Call configure_model before calling configure_optimizers"
        params = _extract_non_bias_params(self.model)
        optimizer = self.optim_cls(
            params,
            **self.optim_conf,
        )
        return optimizer
