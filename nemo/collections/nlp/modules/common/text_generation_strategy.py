from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
import torch
import abc
from typing import List, Tuple

try:
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class TextGenerationStrategy:
    """
    Base class for TextGeneration Strategy
    """
    def __init__(self, model):
        self.model = model
        self.forward_model = self.model.model

    def forward_step(self, batch, tensor_shape):

        if self.model.cfg.get('pipeline_model_parallel_size', 1) > 1:
            output_tensor = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.model.get_forward_output_only_func(),
                batch=batch,
                model=self.forward_model,
                forward_only=True,
                tensor_shape=tensor_shape,
                dtype=self.model.autocast_dtype,
            )
        else:
            output_tensor = forward_backward_no_pipelining(
                forward_step_func=self.model.get_forward_output_only_func(),
                batch=batch,
                model=self.forward_model,
                forward_only=True,
                tensor_shape=tensor_shape,
                dtype=self.model.autocast_dtype,
            )
        return output_tensor

    @abc.abstractclassmethod
    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length
        Args:
            maxlen (int): the max len computed from the context and number of tokens to generate
        returns (int):
            the clip the max length based of the LM model max sequence length
        """
        pass

    @abc.abstractclassmethod
    def init_batch(self, context_tokens: torch.Tensor):
        """initialize the batch data before the inference steps.
           It will save the intermediate results as object attributes
        Args:
            context_tokens (torch.Tensor):  The padded context tokens including the space for tokens to be generated 
        """
        pass

    @abc.abstractclassmethod
    def prepare_batch_at_step(self, tokens: torch.Tensor, maxlen: int, micro_batch_size: int, step: int, context_length: int)->Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps
        Args:
            tokens  (torch.Tensor): the context tokens
            maxlen (int): the maximum length in the context tokens
            micro_batch_size (int): text generation batch size
            step (int): the inference step count
            context_length (int): the new token position in the tokens
        returns:
            a tuple of list of tensor arguments for the model and a list of tensor shape required by forward method
        """
        pass

    @abc.abstractclassmethod
    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the inference, post process the inference results
        Args:
            tokens  (torch.Tensor): the context tokens
            new_token (torch.Tensor): sampled new token id
            context_length (int): the new token position in the tokens
        """
        pass

class GPTModelTextGenerationStrategy(TextGenerationStrategy):

    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length"""
        if maxlen > self.model.cfg.encoder_seq_length + 1:
            maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen
 
    def init_batch(self, context_tokens: torch.Tensor):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()
        # Get the attention mask and postition ids.
        self.attention_mask, _, self.position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eos_id,
            self.model.cfg.get('reset_position_ids', False),
            self.model.cfg.get('reset_attention_mask', False),
            self.model.cfg.get('eod_mask_loss', False),
        )

    def prepare_batch_at_step(self, tokens: torch.Tensor, maxlen: int, micro_batch_size: int, step: int, context_length: int)->Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps
        """
        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :context_length]
            positions2use = self.position_ids[:, :context_length]
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, :context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, context_length - 1].view(micro_batch_size, -1)
            # not using type2use. uncomment it if it is used
                # if type_ids is not None:
                #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)

        """Prepare batch for each of the inference steps"""
        attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, attention_mask_repeat, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape

class PromptLearningModelTextGenerationStrategy(GPTModelTextGenerationStrategy):

    def __init__(self, model, task_ids):
        super(model)
        self.task_ids = task_ids
        self.forward_model = self.model

    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length"""
        if maxlen > self.model.frozen_model.cfg.encoder_seq_length + 1:
            maxlen = self.model.frozen_model.cfg.encoder_seq_length + 1
        return maxlen
 
    def prepare_batch_at_step(self, tokens: torch.Tensor, maxlen: int, micro_batch_size: int, step: int, context_length: int)->Tuple[List[torch.Tensor], List[int]]:
        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :context_length]
            positions2use = self.position_ids[:, :context_length]
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, :context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, context_length - 1].view(micro_batch_size, -1)
            # not using type2use. uncomment it if it is used
                # if type_ids is not None:
                #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)

        """Prepare batch for each of the inference steps"""
        attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, attention_mask_repeat, positions2use, self.task_ids, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.frozen_model.cfg.hidden_size]
        return batch, tensor_shape

    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the inference, post process the inference results
        """
        # Replace special soft prompt token ids with unk token ids
        if (
            self.model.pseudo_token_ids_start is not None
        ):  # TODO: (@adithyare) prompt learning logic can be greatly simplified by removing data preparation logic from model logic.
            tokenizer = self.model.tokenizer
            pseudo_token_ids_start = self.model.pseudo_token_ids_start
            new_tokens[(new_tokens >= pseudo_token_ids_start)] = tokenizer.unk_id
            tokens[:, :context_length][
                (tokens[:, :context_length] >= pseudo_token_ids_start)
            ] = tokenizer.unk_id

class RetroModelTextGenerationStrategy(TextGenerationStrategy):
    pass



def model_inference_strategy_dispatcher(model, **args):
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import MegatronGPTPromptLearningModel
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel

    if isinstance(model, MegatronGPTPromptLearningModel):
        return PromptLearningModelTextGenerationStrategy(model, **args)
    if isinstance(model, MegatronGPTModel):
        return GPTModelTextGenerationStrategy(model)
    elif isinstance(model, MegatronRetrievalModel):
        return RetroModelTextGenerationStrategy(model, **args)
    else:
        raise ValueError(f'{model} is not supported for inference')

    # Should call GPTModel or Megatron Retrieval Model's forward method
 