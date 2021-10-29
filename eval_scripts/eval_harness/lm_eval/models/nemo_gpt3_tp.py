from logging import disable
import os

import transformers
from lm_eval.base import LM
from lm_eval import utils
import sys
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import re
from argparse import ArgumentParser
from pytorch_lightning.trainer.trainer import Trainer
from hydra import compose, initialize
from omegaconf import OmegaConf
import hydra
from torch.utils.data import Dataset, DataLoader

try:
    import nemo.collections.nlp as nemo_nlp
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.asr.models import EncDecCTCModel
    from nemo.collections.nlp.modules.common.megatron.utils import (
        average_losses_across_data_parallel_group,
        get_ltor_masks_and_position_ids,
    )
    from nemo.utils.app_state import AppState
    from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
    from nemo.utils import logging
    from nemo.utils.get_rank import is_global_rank_zero
    from apex.transformer import tensor_parallel, parallel_state
except ModuleNotFoundError:
    print("Importing NeMo module failed, checkout the NeMo submodule")

from .nemo_gpt3 import CustomSaveRestoreConnector


class CustomNLPDDPPlugin(NLPDDPPlugin):
    def setup_distributed(self, global_rank=None, world_size=None):
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.data_parallel_group is None:
            super().setup_distributed(global_rank, world_size)


class RequestDataset(Dataset):
    def __init__(self, model, tokens):
        self.model = model
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        model = self.model
        tokens = self.tokens[i]
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=model.tokenizer.eos_id,
            reset_position_ids=model.cfg.get('reset_position_ids', False),
            reset_attention_mask=model.cfg.get('reset_attention_mask', False),
            eod_mask_loss=model.cfg.get('eod_mask_loss', False),
        )
        return tokens[0], position_ids[0], attention_mask[0]


class CustomModel(MegatronGPTModel):
    def predict_prep(self, inplens, conts):
        self.inplens = inplens
        self.conts = conts

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        tokens, position_ids, attention_mask = batch
        inplens = self.inplens[batch_idx]
        conts = self.conts[batch_idx]
        batch_size = len(tokens)

        output_tensor = self(tokens, position_ids, attention_mask, labels=None)
        # torch.distributed.barrier()
        maybe_multi_logits = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)

        if is_global_rank_zero():
            res = []
            for maybe_logits, inplen, cont_toks in zip(maybe_multi_logits, inplens, conts):
                contlen = len(cont_toks)
                logprobs = F.log_softmax(maybe_logits, dim=1).cpu()
                logprobs = logprobs[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]

                greedy_tokens = logprobs.argmax(dim=-1)

                # cont_toks :: [1, seq]
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                max_equal = (greedy_tokens == cont_toks).all()

                # last_token_slice = logprobs[:, -1, :].squeeze(0).tolist()

                logprobs = torch.gather(logprobs, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                greedy_tokens = self.tokenizer.ids_to_tokens(
                    greedy_tokens.cpu().numpy().tolist()[0])
                cont_toks = self.tokenizer.ids_to_tokens(cont_toks.cpu().numpy().tolist()[0])
                answer = (float(logprobs.sum()), bool(max_equal), greedy_tokens, cont_toks)
                res.append(answer)
        else:
            res = None
        return res


def setup_trainer_and_model(args):
    """Setup model and optimizer."""
    torch.set_grad_enabled(False)

    assert 'nemo_model' in args, "Path to model's .nemo file is required."
    if 'tensor_model_parallel_size' not in args:
        args['tensor_model_parallel_size'] = 1
    else:
        args['tensor_model_parallel_size'] = int(args['tensor_model_parallel_size'])

    logging.info(f"**** Loading checkpoint from {args['nemo_model']}")
    vocab_file = args.get('vocab_file', None)
    merge_file = args.get('merge_file', None)
    # if args['nemo_model'].rstrip().endswith(".nemo"):
    trainer = Trainer(
        plugins=CustomNLPDDPPlugin(), gpus=args['tensor_model_parallel_size'], accelerator="ddp",
        enable_progress_bar=True)
    app_state = AppState()
    app_state.model_parallel_size = args['tensor_model_parallel_size']
    app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)
    model = CustomModel.restore_from(
        restore_path=args['nemo_model'], trainer=trainer, save_restore_connector=CustomSaveRestoreConnector(
            vocab_file=vocab_file, merge_file=merge_file))

    return trainer, model


def unbatch(outputs):
    res = []
    for batch in outputs:
        for item in batch:
            res.append(item)
    return res


class NeMo_GPT3LM_TP(LM):
    MAX_LENGTH = 2048
    MAX_GEN_TOKS = 256

    def __init__(self, args, device=None, truncate=False, batch_size=1):
        super().__init__()

        # get megatron
        logging.info(f'**** Building GPT model ...')
        self.trainer, self.gpt3 = setup_trainer_and_model(args)
        self.tokenizer = self.gpt3.tokenizer
        self.gpt3.eval()

        self.max_length = self.gpt3.cfg.get('max_position_embeddings')
        assert self.tokenizer.text_to_ids('hello\n\nhello') == [31373, 198, 198, 31373]

        self.truncate = truncate
        self.batch_size = batch_size

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config={}):
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(args, **args2)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [50256]
            else:
                context_enc = self.tokenizer.text_to_ids(context)

            continuation_enc = self.tokenizer.text_to_ids(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        len_rolling_token_windows = [0]
        all_rolling_token_windows = []
        with torch.no_grad():

            for string, in requests:
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer.text_to_ids(string),
                    prefix_token=50256,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]
                len_rolling_token_windows.append(len(rolling_token_windows) + len_rolling_token_windows[-1])
                all_rolling_token_windows.extend(rolling_token_windows)

            # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
            string_nll = self._loglikelihood_tokens(all_rolling_token_windows)
            if self.can_access_output():
                string_nll = [x[0] for x in string_nll]
                # discard is_greedy
                for i in range(len(len_rolling_token_windows) - 1):
                    loglikelihoods.append(
                        sum(string_nll[len_rolling_token_windows[i]:len_rolling_token_windows[i + 1]]))

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        with torch.no_grad():

            def _collate(x):
                # the negative sign on len(toks) sorts descending - this has a few advantages:
                # - time estimates will always be over not underestimates, which is more useful for planning
                # - to know the size of a batch when going through the list, you know the first one is always the batch padded context length.
                #   this is useful to simplify the batching logic and more importantly to make automatic adaptive batches much much easier to implement
                # - any OOMs will happen right away rather than near the end

                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            # TODO: automatic (variable) batch size detection for vectorization
            reord = utils.Reorderer(requests, _collate)

            # for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm),
            #                           n=self.batch_size):  # NOTE: hard-code batch size to be 1 for 530B model for now

            inps = []
            all_conts = []
            all_inplens = []
            for chunk in utils.chunks(reord.get_reordered(), n=self.batch_size):
                padding_length = None
                # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
                # tensors, then we pack them together into a batch, call the model, and then pick it all apart
                # again because vectorizing is annoying
                conts = []
                inplens = []
                for _, context_enc, continuation_enc in chunk:
                    # sanity check
                    assert len(context_enc) > 0
                    assert len(continuation_enc) > 0
                    assert len(continuation_enc) <= self.max_length

                    # how this all works:
                    #          CTX      CONT
                    # inp    0 1 2 3|4 5 6 7 8 9 <- last token is deleted by inp[:, :-1]
                    # gpt2    \               \
                    # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the [:, -len(continuation_enc):, :self.VOCAB_SIZE] slice
                    # cont_toks      4 5 6 7 8 9

                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))
                    conts.append(cont)
                    inplens.append(inplen)
                all_inplens.append(inplens)
                all_conts.append(conts)

            self.gpt3.predict_prep(all_inplens, all_conts)
            request_ds = RequestDataset(self.gpt3, inps)
            request_dl = DataLoader(request_ds, batch_size=self.batch_size)

            res = self.trainer.predict(self.gpt3, request_dl)

        return reord.get_original(unbatch(res)) if self.can_access_output() else None

    def greedy_until(self, requests):
        raise NotImplementedError

    def can_access_output(self):
        return is_global_rank_zero()
