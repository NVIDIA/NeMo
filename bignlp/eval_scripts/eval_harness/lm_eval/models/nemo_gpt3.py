import os
import tempfile

from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm
import torch
import torch.nn.functional as F

from pytorch_lightning.trainer.trainer import Trainer
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict

import hydra

import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_ltor_masks_and_position_ids,
)
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import logging
from nemo.utils.app_state import AppState


class CustomSaveRestoreConnector(SaveRestoreConnector):
    def __init__(self, merge_file=None, vocab_file=None):
        super().__init__()
        self.merge_file = merge_file
        self.vocab_file = vocab_file

    def restore_from(
        self,
        calling_cls,
        restore_path: str,
        override_config_path=None,
        map_location=None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        """
        Restores model instance (weights and configuration) into .nemo file

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict. By default True
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.

        Example:
            ```
            model = nemo.collections.asr.models.EncDecCTCModel.restore_from('asr.nemo')
            assert isinstance(model, nemo.collections.asr.models.EncDecCTCModel)
            ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        cwd = os.getcwd()

        if map_location is None:
            if torch.cuda.is_available():
                map_location = torch.device('cuda')
            else:
                map_location = torch.device('cpu')

        app_state = AppState()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                self._unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                else:
                    # can be str path or OmegaConf / DictConfig object
                    config_yaml = override_config_path
                if not isinstance(config_yaml, (OmegaConf, DictConfig)):
                    conf = OmegaConf.load(config_yaml)
                else:
                    conf = config_yaml
                    if override_config_path is not None:
                        # Resolve the override config
                        conf = OmegaConf.to_container(conf, resolve=True)
                        conf = OmegaConf.create(conf)
                # If override is top level config, extract just `model` from it
                if 'model' in conf:
                    conf = conf.model

                if return_config:
                    instance = conf
                    return instance
                else:
                    if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                        model_weights = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.model_weights_ckpt)
                    else:
                        model_weights = os.path.join(tmpdir, self.model_weights_ckpt)
                OmegaConf.set_struct(conf, True)
                with open_dict(conf):
                    conf.precision = "bf16"
                    conf.megatron_amp_O2 = False
                os.chdir(cwd)
                # get the class
                calling_cls._set_model_restore_state(is_being_restored=True, folder=tmpdir)

                # if "precision" in conf:
                #     conf.precision = 16
                # if "megatron_amp_O2" in conf:
                #     conf.megatron_amp_O2 = False

                if self.vocab_file is not None:
                    conf.tokenizer.vocab_file = self.vocab_file
                if self.merge_file is not None:
                    conf.tokenizer.merge_file = self.merge_file
                instance = calling_cls.from_config_dict(config=conf, trainer=trainer)
                instance = instance.to(map_location)
                # add load_state_dict override
                if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                    model_weights = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.model_weights_ckpt)
                state_dict = self._load_state_dict_from_disk(model_weights, map_location=map_location)
                if conf.get('megatron_amp_O2', False):
                    new_state_dict = {}
                    for key in state_dict.keys():
                        new_key = key.replace('model.', 'model.module.', 1)
                        new_state_dict[new_key] = state_dict[key]
                    state_dict = new_state_dict
                instance.load_state_dict(state_dict, strict=strict)

                logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')
                instance._set_model_restore_state(is_being_restored=False)
            finally:
                os.chdir(cwd)

        return instance


def setup_model(args):
    """Setup model and optimizer."""
    torch.set_grad_enabled(False)
    trainer = Trainer(gpus=1)
    assert 'nemo_model' in args, "Path to model's .nemo file is required."
    if args['nemo_model'] is not None:
        logging.info(f"**** Loading checkpoint from {args['nemo_model']}")
        vocab_file = args.get('vocab_file', None)
        merge_file = args.get('merge_file', None)
        if args['nemo_model'].rstrip().endswith(".nemo"):
            model = MegatronGPTModel.restore_from(
                restore_path=args['nemo_model'], trainer=trainer, save_restore_connector=CustomSaveRestoreConnector(
                    vocab_file=vocab_file, merge_file=merge_file))
        elif args['nemo_model'].rstrip().endswith(".ckpt"):
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            initialize(config_path="conf/nemo-nlp-conf", job_name="eval")
            cfg = compose(config_name="megatron_gpt_config", overrides=[])

            def update(d_from, d_to):
                for k, v in d_from.items():
                    if k in d_to:
                        d_to[k] = v

            load = torch.load(args['nemo_model'], map_location="cpu")
            update(load["hyper_parameters"], cfg.model)
            # TODO: force fp16 to be force before fp16 supported in evaluation
            cfg.model['fused_fp16'] = False
            cfg.model['fused_bf16'] = False

            # Create an empty model with hyperparameters from loaded checkpoint
            model = MegatronGPTModel(cfg.model, trainer)
            model.load_state_dict(load['state_dict'])
            model = model.cuda()
        else:
            raise ValueError(f"Invalid checkpoint type from {args['nemo_model'].strip().split('/')[-1]}.")
    else:
        logging.info(f"**** NOT Loading checkpoint; starting a scratch model")
        raise NotImplementedError
        # model = MegatronGPTModel(args.cfg, trainer)

    return model


class NeMo_GPT3LM(LM):
    MAX_LENGTH = 2048
    MAX_GEN_TOKS = 256

    def __init__(self, args, device=None, truncate=False, batch_size=1):
        super().__init__()

        # get megatron
        logging.info(f'**** Building GPT model ...')
        self.gpt3 = setup_model(args)
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
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer.text_to_ids(string),
                    prefix_token=50256,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def get_batch(self, context_tokens):
        model = self.gpt3
        tokens = context_tokens
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=model.tokenizer.eos_id,
            reset_position_ids=model.cfg.get('reset_position_ids', False),
            reset_attention_mask=model.cfg.get('reset_attention_mask', False),
            eod_mask_loss=model.cfg.get('eod_mask_loss', False),
        )
        return tokens, attention_mask, position_ids

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

            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm),
                                      n=self.batch_size):  # NOTE: hard-code batch size to be 1 for 530B model for now
                inps = []
                conts = []
                inplens = []

                padding_length = None
                # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
                # tensors, then we pack them together into a batch, call the model, and then pick it all apart
                # again because vectorizing is annoying

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
                        , dtype=torch.long).cuda()
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).cuda()  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))
                    conts.append(cont)
                    inplens.append(inplen)

                maybe_multi_logits = self._model_call_megatron(torch.cat(inps, dim=0))  # [batch, seq, vocab]

                for (cache_key, _, _), maybe_logits, inp, inplen, cont_toks in zip(chunk, maybe_multi_logits, inps,
                                                                                   inplens, conts):
                    contlen = len(cont_toks)

                    if self.can_access_output():
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
                    else:
                        answer = None

                    # partial caching
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                    res.append(answer)

        print(reord.get_original(res))
        return reord.get_original(res)

    def _model_call_megatron(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits retuned from the model
        """
        _, attention_mask, position_ids = self.get_batch(inps)

        with torch.no_grad():
            output_tensor = self.gpt3(inps, position_ids, attention_mask, labels=None)

        # output_tensor = output_tensor[..., :50257].contiguous()
        ret_output_tensor = output_tensor[:, :, :50257]
        return ret_output_tensor

    def greedy_until(self, requests):
        raise NotImplementedError

    def can_access_output(self):
        return True
