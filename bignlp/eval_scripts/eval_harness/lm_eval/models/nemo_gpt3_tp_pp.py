# This model is for debugging only!
from lm_eval.base import LM
from lm_eval import utils
import os
import torch
import logging
import torch.nn.functional as F
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import nemo.collections.nlp as nemo_nlp
from torch.nn.utils.rnn import pad_sequence
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    fake_initialize_model_parallel,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_ltor_masks_and_position_ids,
)
from nemo.utils.app_state import AppState
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.model_utils import inject_model_parallel_rank


from apex.transformer import tensor_parallel, parallel_state
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
    forward_backward_pipelining_without_interleaving,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import (
    forward_backward_no_pipelining,
)
from apex.transformer.pipeline_parallel.utils import (
    get_num_microbatches,
    _reconfigure_microbatch_calculator,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s | %(name)-7s | %(levelname)-8s: %(message)s"
)
logger = logging.getLogger(__name__)


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
        continuation_enc = (
            self.tokenizer.text_to_ids(continuation)
            if isinstance(continuation, str)
            else continuation
        )
        # sanity check
        assert len(context_enc) > 0
        assert len(continuation_enc) > 0
        assert len(continuation_enc) <= self.max_length

        conti_len = len(continuation_enc)
        inp_enc = torch.tensor((context_enc + continuation_enc)[-(self.max_length + 1) :])
        return inp_enc, conti_len


class EvalGPTModel(MegatronGPTModel):
    @classmethod
    def _bucketize_gpt_inference(cls, batch, use_soft_prompts=False):
        batch_tokens, lens, tokens_to_generate, compute_logprobs = batch[:4]
        batch_size = len(batch_tokens)
        tokens_to_generate = tokens_to_generate[0]
        batch_tokens = batch_tokens.tolist()

        if use_soft_prompts:
            prompt_tags = batch[4]

        # unpad tokens
        indxs = [index for index in range(batch_size)]
        for lenn, index in zip(lens, indxs):
            batch_tokens[index] = batch_tokens[index][:lenn]

        # chunk tokens by same length
        pre_buckets, lens = [], list(set(lens.tolist()))
        for lenn in lens:
            pre_buckets.append(
                [
                    (tokens, index)
                    for index, tokens in enumerate(batch_tokens)
                    if len(tokens) == lenn
                ]
            )

        buckets, positions, bucket_prompt_tags = [], [], []

        # get buckets and prompts initial positions
        for bucket in pre_buckets:
            buckets.append(torch.tensor([item[0] for item in bucket]).to(device="cuda"))
            positions.append([item[1] for item in bucket])

            # bucket prompt tags identically to their corresponding examples
            if use_soft_prompts:
                bucket_prompt_tags.append([prompt_tags[item[1]] for item in bucket])

        # Flatten position list
        positions = [item for sublist in positions for item in sublist]

        # Form request
        request = {"tokens": buckets, "prompt_tags": bucket_prompt_tags}

        return request, positions, tokens_to_generate, compute_logprobs[0]

    def compute_logprobs(self, request, positions):
        """
            Only logprobs computation without generation tokens
        Args:
            request:
                * tokens: List of "buckets" with unpadded tokens of the same length
                * prompt_tags: List of "buckets" where each bucket contains the prompt_tag strings
                                    specifying the prompt tag to use (optional)
            positions: List with initial prompts positions
        Returns:
            response: A python list of tuples
            (text, tokens, log_probs, offsets)
            * text: string, inputted prompt + generated text by model
            * tokens: list of tokens correspond to text
            * log_probs: list of log_softmax's from output_tensor in respect to text tokens
            * offsets: list of tokens start positions in text
        """
        app_state = AppState()

        results = []
        request_tokens = request["tokens"]
        for idx, tokens in enumerate(request_tokens):
            tokens_cut = tokens[:, :-1]
            micro_batch_size = tokens_cut.shape[0]
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=micro_batch_size,
                micro_batch_size=micro_batch_size,
                data_parallel_size=1,
            )

            # Force soft prompts to be false
            self.use_soft_prompts = False
            # For prompt tuned GPT models
            if self.use_soft_prompts:
                if self.cfg.get("pipeline_model_parallel_size", 1) > 1:
                    raise ValueError(
                        "compute_logprobs method is not yet supported for pipeline with soft prompts"
                    )
                prompt_tags = request["prompt_tags"][idx]
                prompt_tags_to_ids = dict(self.prompt_table)
                prompt_ids = torch.tensor([prompt_tags_to_ids[tag] for tag in prompt_tags])
            else:
                prompt_ids = None

            if self.use_soft_prompts:
                batch_size = len(tokens_cut)
                full_length = len(tokens_cut[0]) + self.num_prompt_tokens
                # Get postion ids for text after soft prompt
                position_ids = torch.arange(
                    start=self.num_prompt_tokens,
                    end=full_length,
                    dtype=torch.long,
                    device=self.device,
                )
                position_ids = position_ids.unsqueeze(0).expand_as(tokens_cut).clone()
                # Make attention mask starting with first token in soft prompt
                attention_mask = torch.tril(
                    torch.ones((batch_size, full_length, full_length), device=self.device)
                ).view(batch_size, 1, full_length, full_length)
                attention_mask = attention_mask < 0.5

            else:
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=tokens_cut,
                    eod_token=self.tokenizer.eos_id,
                    reset_position_ids=self.cfg.get("reset_position_ids", False),
                    reset_attention_mask=self.cfg.get("reset_attention_mask", False),
                    eod_mask_loss=self.cfg.get("eod_mask_loss", False),
                )

            # we repeat attention mask to work with apex fwd/bwd function
            attention_mask_repeat = torch.concat([attention_mask for _ in range(micro_batch_size)])
            if self.use_soft_prompts:
                batch = [tokens_cut, attention_mask_repeat, position_ids, prompt_ids]
            else:
                batch = [tokens_cut, attention_mask_repeat, position_ids]
            tensor_shape = [tokens_cut.shape[1], micro_batch_size, self.cfg.hidden_size]
            if self.cfg.get("pipeline_model_parallel_size", 1) > 1:
                output_tensor = forward_backward_pipelining_without_interleaving(
                    forward_step_func=self.get_forward_output_only_func(),
                    batch=batch,
                    model=self.model,
                    forward_only=True,
                    tensor_shape=tensor_shape,
                    dtype=self.autocast_dtype,
                )
            else:
                output_tensor = forward_backward_no_pipelining(
                    forward_step_func=self.get_forward_output_only_func(),
                    batch=batch,
                    model=self.model,
                    forward_only=True,
                    tensor_shape=tensor_shape,
                    dtype=self.autocast_dtype,
                )

            # get output tensor
            if parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor[0]["logits"]
                output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(
                    output_tensor
                )

            else:
                output_tensor = torch.zeros(
                    (tokens_cut.shape[0], tokens_cut.shape[1], self.padded_vocab_size),
                    dtype=torch.float,
                ).cuda()

            torch.distributed.broadcast(output_tensor, get_last_rank())

            log_probs = []
            for output in output_tensor:
                probs = F.log_softmax(output, dim=1)
                probs = probs[-len(tokens_cut[0]) :]
                log_probs.append(probs)

            for token, prob in zip(tokens, log_probs):
                results.append(
                    (
                        self.tokenizer.ids_to_text(token),
                        self.tokenizer.ids_to_tokens(token),
                        prob,
                        [0],
                    )
                )

        # offsets calculation
        for item in results:
            for index, token in enumerate(item[1]):
                if index != len(item[1]) - 1:
                    item[3].append(len(token) + item[3][-1])

        # return prompts in order they were inputted
        response = [0 for i in range(len(positions))]
        for item, index in zip(results, positions):
            response[index] = item

        return response

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        (
            request,
            positions,
            tokens_to_generate,
            compute_logprobs,
        ) = EvalGPTModel._bucketize_gpt_inference(batch, False)
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
                greedy_tokens = self.tokenizer.ids_to_tokens(greedy_tokens.cpu().numpy().tolist())

                conti_tokens = inp_tokens[-conti_len:]
                # conti_tokens = self.tokenizer.ids_to_tokens(conti_tokens.cpu().numpy().tolist())
                max_equal = greedy_tokens == conti_tokens

                log_probs = log_probs.cpu().to(torch.float32)
                conti_enc = torch.tensor(self.tokenizer.tokens_to_ids(conti_tokens))
                conti_probs = torch.gather(log_probs, 1, conti_enc.unsqueeze(-1)).squeeze(-1)

                res.append((float(conti_probs.sum()), bool(max_equal), greedy_tokens, conti_tokens))
            return res

        return None


def setup_trainer_and_model(args):
    """Setup model and optimizer."""
    torch.set_grad_enabled(False)

    assert args.nemo_model is not None or (
        args.checkpoint_folder is not None and args.checkpoint_name is not None
    ), "Path to checkpoints is required."

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
        app_state.model_parallel_size = (
            args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        )
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            _,
            _,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=args.tensor_model_parallel_size,
            pipeline_model_parallel_size_=args.pipeline_model_parallel_size,
        )

    if args.nemo_model is not None:
        logger.info(f"**** Loading checkpoint from {args.nemo_model}")
        model = EvalGPTModel.restore_from(restore_path=args.nemo_model, trainer=trainer)
    else:
        if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
            app_state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = args.tensor_model_parallel_size

        logger.info(
            f"**** Loading checkpoint from {args.checkpoint_folder} - {args.checkpoint_name}"
        )
        # inject model parallel rank
        checkpoint_path = inject_model_parallel_rank(
            os.path.join(args.checkpoint_folder, args.checkpoint_name)
        )
        model = EvalGPTModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )

    model.freeze()

    return trainer, model


class NeMo_GPT3LM_TP_PP(LM):
    def __init__(self, args, truncate=False, batch_size=1):
        super().__init__()

        # get megatron
        logger.info(f"**** Building GPT model ...")
        self.trainer, self.gpt3 = setup_trainer_and_model(args)
        self.tokenizer = self.gpt3.tokenizer
        self.gpt3.eval()

        self.max_length = self.gpt3.cfg.get("max_position_embeddings")
        assert self.tokenizer.text_to_ids("hello\n\nhello") == [31373, 198, 198, 31373]

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
                new_batch.append(
                    (token, lenn, 16, True, conti_len)
                )  # 16 is a placeholder and never been used

            new_batch = default_collate(new_batch)
            return new_batch

        def _collate(x):  # used to reorder request and remove duplications
            toks = x[0] + x[1]
            return -len(toks), tuple(toks)

        reord = utils.Reorderer(requests, _collate)
        request_ds = RequestDataset(reord.get_reordered(), self.gpt3.tokenizer)
        # request_ds = RequestDataset(requests, self.gpt3.tokenizer)
        request_dl = DataLoader(
            request_ds, collate_fn=pad_collate, batch_size=self.batch_size, shuffle=False
        )
        res = self.trainer.predict(self.gpt3, request_dl)
        return (
            reord.get_original([item for batch in res for item in batch])
            if self.can_access_output()
            else None
        )

    def loglikelihood_rolling(self, requests):
        loglikelihoods = []
        len_rolling_token_windows = [0]
        all_rolling_token_windows = []

        for (string,) in requests:
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tokenizer.text_to_ids(string),
                        prefix_token=50256,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            len_rolling_token_windows.append(
                len(rolling_token_windows) + len_rolling_token_windows[-1]
            )
            all_rolling_token_windows.extend(rolling_token_windows)

        string_nll = self._loglikelihood(all_rolling_token_windows)
        if self.can_access_output():
            string_nll = [x[0] for x in string_nll]
            # discard is_greedy
            for i in range(len(len_rolling_token_windows) - 1):
                loglikelihoods.append(
                    sum(string_nll[len_rolling_token_windows[i] : len_rolling_token_windows[i + 1]])
                )

        return loglikelihoods

    def greedy_until(self, requests):
        raise NotImplementedError

    def can_access_output(self):
        return is_global_rank_zero()
