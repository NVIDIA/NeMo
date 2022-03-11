from lm_eval.base import LM
from lm_eval import utils
import os
import torch
import torch.nn.functional as F
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import nemo.collections.nlp as nemo_nlp
from torch.nn.utils.rnn import pad_sequence
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_ltor_masks_and_position_ids,
)
from nemo.utils.app_state import AppState
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.model_utils import inject_model_parallel_rank

from apex.transformer import tensor_parallel, parallel_state


class RequestDataset(Dataset):
    def __init__(self, requests, tokenizer) -> None:
        super().__init__()
        self.requests = requests
        self.tokenizer = tokenizer
        self.max_length = 2048

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, index):
        context, continuation = self.requests[index]
        context_enc = self.tokenizer.text_to_ids(context) if isinstance(context, str) else context
        continuation_enc = self.tokenizer.text_to_ids(continuation) if isinstance(continuation, str) else continuation
        # sanity check
        assert len(context_enc) > 0
        assert len(continuation_enc) > 0
        assert len(continuation_enc) <= self.max_length

        conti_len = len(continuation_enc)
        inp_enc = torch.tensor((context_enc + continuation_enc)[-(self.max_length + 1):])
        return inp_enc, conti_len


class EvalGPTModel(MegatronGPTModel):
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        request, positions, tokens_to_generate, compute_logprobs = MegatronGPTModel._bucketize_gpt_inference(
            batch, False
        )
        response = self.compute_logprobs(request, positions)
        if is_global_rank_zero():
            _, lens, _, _, conti_lens = batch
            batch_size = len(lens)
            assert len(response) == batch_size, "Response's length not equal to batch size."
            res = []
            for index in range(batch_size):
                conti_len = conti_lens[index]

                inp_tokens = response[index][1]
                assert len(inp_tokens) == lens[index], "Mismatch in input tokens length."

                log_probs = response[index][2]
                log_probs = log_probs[-conti_len:]

                greedy_tokens = log_probs.argmax(dim=-1)
                greedy_tokens = self.tokenizer.ids_to_tokens(
                    greedy_tokens.cpu().numpy().tolist())

                conti_tokens = inp_tokens[-conti_len:]
                # conti_tokens = self.tokenizer.ids_to_tokens(conti_tokens.cpu().numpy().tolist())
                max_equal = (greedy_tokens == conti_tokens)

                log_probs = log_probs.cpu().to(torch.float32)
                conti_enc = torch.tensor(self.tokenizer.tokens_to_ids(conti_tokens))
                conti_probs = torch.gather(log_probs, 1, conti_enc.unsqueeze(-1)).squeeze(-1)

                res.append((float(conti_probs.sum()), bool(max_equal), greedy_tokens, conti_tokens))
            return res

        return None


def setup_trainer_and_model(args):
    """Setup model and optimizer."""
    torch.set_grad_enabled(False)

    assert args.nemo_model is not None or \
           (args.checkpoint_folder is not None and args.checkpoint_name is not None), "Path to checkpoints is required."

    # cast precision to int if 32 or 16
    if args.precision in ["32", "16"]:
        args.precision = int(float(args.precision))

    model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    num_nodes = max(model_parallel_size // 8, 1)
    gpus = min(model_parallel_size, 8)

    trainer = Trainer(
        plugins=NLPDDPPlugin(),
        gpus=gpus,
        num_nodes=num_nodes,
        precision=args.precision,
    )

    app_state = AppState()
    if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            _,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=args.tensor_model_parallel_size,
            pipeline_model_parallel_size_=args.pipeline_model_parallel_size,
        )

    if args.nemo_model is not None:
        logging.info(f"**** Loading checkpoint from {args.nemo_model}")
        model = EvalGPTModel.restore_from(
            restore_path=args.nemo_model, trainer=trainer
        )
    else:
        if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
            app_state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = args.tensor_model_parallel_size

        logging.info(f"**** Loading checkpoint from {args.checkpoint_folder} - {args.checkpoint_name}")
        # inject model parallel rank
        checkpoint_path = inject_model_parallel_rank(os.path.join(args.checkpoint_folder, args.checkpoint_name))
        model = EvalGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)
        
    model.freeze()

    return trainer, model


class NeMo_GPT3LM_TP_PP(LM):
    def __init__(self, args, truncate=False, batch_size=1):
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
        return self._loglikelihood(requests)

    # request: (context, continuation)
    def _loglikelihood(self, requests):
        def pad_collate(batch):
            tokens = [item[0] for item in batch]
            conti_lens = [item[1] for item in batch]
            lens = [len(token) for token in tokens]

            tokens_pad = pad_sequence(tokens, batch_first=False, padding_value=50256)
            new_batch = []

            for token, lenn, conti_len in zip(tokens_pad.T, lens, conti_lens):
                # (token, lenn, tokens_to_generate, compute_logprobs)
                new_batch.append((token, lenn, 16, True, conti_len))  # 16 is a placeholder and never been used

            new_batch = default_collate(new_batch)
            return new_batch

        def _collate(x):  # used to reorder request and remove duplications
            toks = x[0] + x[1]
            return -len(toks), tuple(toks)

        reord = utils.Reorderer(requests, _collate)
        request_ds = RequestDataset(reord.get_reordered(), self.gpt3.tokenizer)
        # request_ds = RequestDataset(requests, self.gpt3.tokenizer)
        request_dl = DataLoader(request_ds, collate_fn=pad_collate, batch_size=self.batch_size, shuffle=False)
        res = self.trainer.predict(self.gpt3, request_dl)
        return reord.get_original([item for batch in res for item in batch]) if self.can_access_output() else None

    def loglikelihood_rolling(self, requests):
        loglikelihoods = []
        len_rolling_token_windows = [0]
        all_rolling_token_windows = []

        for string, in requests:
            rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                token_list=self.tokenizer.text_to_ids(string),
                prefix_token=50256,
                max_seq_len=self.max_length,
                context_len=1,
            )))

            len_rolling_token_windows.append(len(rolling_token_windows) + len_rolling_token_windows[-1])
            all_rolling_token_windows.extend(rolling_token_windows)

        string_nll = self._loglikelihood(all_rolling_token_windows)
        if self.can_access_output():
            string_nll = [x[0] for x in string_nll]
            # discard is_greedy
            for i in range(len(len_rolling_token_windows) - 1):
                loglikelihoods.append(
                    sum(string_nll[len_rolling_token_windows[i]:len_rolling_token_windows[i + 1]]))

        return loglikelihoods

    def greedy_until(self, requests):
        raise NotImplementedError

    def can_access_output(self):
        return is_global_rank_zero()
