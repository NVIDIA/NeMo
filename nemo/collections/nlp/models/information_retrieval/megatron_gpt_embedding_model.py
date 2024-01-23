
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
import torch
try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_current_global_batch_size,
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False
    
class MegatronGPTEmbeddingModel(MegatronGPTSFTModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(cfg, trainer=trainer)

    def model_provider_func(self, pre_process, post_process):
        # (@adithyare) We need post_process to be False to get hidden states in the loss_func
        return super().model_provider_func(pre_process, post_process=False)
    
    def loss_func(self, loss_mask, num_valid_tokens_in_ub, output_tensor):
        query_hs = output_tensor[-1, ::3, :]
        pos_doc_hs = output_tensor[-1, 1::3, :]
        neg_doc_hs = output_tensor[-1, 2::3, :]
        pos_cs = torch.nn.functional.cosine_similarity(query_hs, pos_doc_hs, dim=-1).sum()
        neg_cs = torch.nn.functional.cosine_similarity(query_hs, neg_doc_hs, dim=-1).sum()
        print(pos_cs, "pos cs")
        print(neg_cs, "neg cs\n\n")
        loss = pos_cs - neg_cs
        cp_size = self.cfg.get('context_parallel_size', 1)
        if cp_size > 1:
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        return loss