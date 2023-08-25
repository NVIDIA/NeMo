# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import os
import re
from typing import List, Tuple

import torch

from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

try:
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


# the text representation of eos_id, it applies for all tokenizers
END_OF_SEQ = '<|endoftext|>'


class TextGenerationStrategy:
    """
    Base class for TextGeneration Strategy
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward_step(self, batch, tensor_shape):
        fwd_bwd_function = get_forward_backward_func()
        output_tensor = fwd_bwd_function(
            forward_step_func=self.model.get_forward_output_only_func(),
            data_iterator=iter([batch,]),
            model=[self.forward_model],
            num_microbatches=get_num_microbatches(),
            forward_only=True,
            seq_length=tensor_shape[0],
            micro_batch_size=tensor_shape[1],
        )

        return output_tensor

    def tokenize_batch(self, sentences, max_len, add_BOS):
        """
        convert the sentences into lists of tokens, pad them to the same length, add bos tokens if it is needed
        Args:
            sentences (List[str]): list of input sentences in str format.
            max_len (int): max number of tokens to generate.
            add_BOS (bool): whether to add the BOS token at the beginning
        Returns:
            Tuple[torch.Tensor], the tokenized and padded torch tensor and the token context length tensor.
        """
        tokenizer = self.model.tokenizer
        if add_BOS:
            context_tokens = [[tokenizer.bos_id] + tokenizer.text_to_ids(s) for s in sentences]
        else:
            context_tokens = [tokenizer.text_to_ids(s) for s in sentences]
        context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

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
    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps.
           It will save the intermediate results as object attributes
           context_length (int): the context token length
           compute_attention_mask: bool: set to True to compute attention mask (not needed for FA)
        Args:
            context_tokens (torch.Tensor):  The padded context tokens including the space for tokens to be generated 
        """
        pass

    @abc.abstractclassmethod
    def prepare_batch_at_step(
        self, tokens: torch.Tensor, maxlen: int, micro_batch_size: int, step: int, context_length: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
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
        At the end of the single step inference, post process the inference results
        Args:
            tokens  (torch.Tensor): the context tokens
            new_token (torch.Tensor): sampled new token id
            context_length (int): the new token position in the tokens
        """
        pass

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        if len(end_strings) == 1 and end_strings[0] == END_OF_SEQ:
            return prev == eod_id
        else:
            tokenizer = self.model.tokenizer
            conditions = []
            end_tokens = set()
            end_tokens.add(eod_id)
            for end_string in end_strings:
                ids_1 = tokenizer.text_to_ids(f'<extra_id_1>{end_string}')
                ids_2 = tokenizer.text_to_ids('<extra_id_1>')
                if len(ids_1) <= len(ids_2):
                    continue
                token_id = ids_1[len(ids_2) :][0]
                end_tokens.add(token_id)
            for p, token_item in zip(prev, tokens):
                text = tokenizer.ids_to_text(token_item.tolist())
                conditions.append(
                    any([text.endswith(end_string) for end_string in end_strings] + [p.item() in end_tokens])
                )
            return torch.tensor(conditions, dtype=torch.bool, device=tokens.device)

    def post_generation_process(self, output):
        """
        At the end of the text generation, post process the results
        Args:
            output  (dict): the text generation output dictionary
        """
        return output


class GPTModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model):
        super().__init__(model)
        self.forward_model = self.model.model

    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length"""

        # for positional embedding types that allow length extrapolation, don't clip the max length
        if self.model.cfg.get("position_embedding_type", "learned_absolute") == "learned_absolute":
            if maxlen > self.model.cfg.encoder_seq_length + 1:
                maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
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
            compute_attention_mask=compute_attention_mask,
        )

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool = True,
    ) -> Tuple[List[torch.Tensor], List[int]]:
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
        attention_mask_repeat = None
        if compute_attention_mask:
            attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])

        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, attention_mask_repeat, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape


class NevaModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model):
        super().__init__(model)
        self.forward_model = self.model.model.module
        self.num_media_latents = 576  # TODO: Need to obtain this from the config ideally
        self.tokenizer = self.model.tokenizer
        self.image_paths = []
        self.cfg = model.cfg
        self.data_cfg = model.cfg.data
        from transformers import CLIPImageProcessor

        if self.cfg.mm_cfg.vision_encoder.from_hf:
            self.processor = CLIPImageProcessor.from_pretrained(
                self.cfg.mm_cfg.vision_encoder.from_pretrained, torch_dtype=torch.bfloat16
            )
        else:
            self.processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16
            )

        self.model = model

    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length"""
        if maxlen > self.model.cfg.encoder_seq_length + 1:
            maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()
        # Get the attention mask and postition ids.
        self.attention_mask, _, self.position_ids = get_ltor_masks_and_position_ids(
            tokens,
            eod_token=tokenizer.eos_id,
            eod_mask_loss=False,
            reset_attention_mask=False,
            reset_position_ids=False,
            compute_attention_mask=compute_attention_mask,
        )

    def process_prompts(self, prompt):
        from nemo.collections.multimodal.data.neva.neva_dataset import DEFAULT_IMAGE_TOKEN, preprocess

        list_data_dict = []

        record = {
            'system': 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n\n',
            'conversations': [
                {'from': 'User', 'value': prompt,},
                {
                    'from': 'Assistant',
                    'value': '',
                    'label': 'quality:8,toxicity:0,humor:0,creativity:0,violence:0,helpfulness:8,not_appropriate:0',
                },
            ],
        }

        for turn in record['conversations']:  #
            if turn.get('value') is not None:
                turn['value'] = re.sub('<image>', f'{DEFAULT_IMAGE_TOKEN}\n', turn['value'])
        list_data_dict.append(record)

        add_extra_token = 1
        if getattr(self.model.cfg, 'no_seqlen_plus_one_input_tokens', False):
            add_extra_token = 0
        data_cfg = self.model.cfg.data
        model_cfg = self.model.cfg

        multimodal_cfg = dict(
            is_multimodal=data_cfg.is_multimodal,
            sep_image_conv_front=data_cfg.sep_image_conv_front,
            image_token_len=data_cfg.image_token_len,
            image_folder=data_cfg.image_folder,
            image_aspect_ratio=data_cfg.image_aspect_ratio,
            use_im_start_end=getattr(model_cfg.mm_cfg, 'use_im_start_end', False),
            image_processor=self.processor,
            add_extra_token=add_extra_token,
            context_length=model_cfg.encoder_seq_length,
        )

        import copy

        from nemo.collections.multimodal.data.neva.neva_dataset import preprocess_multimodal

        sources = preprocess_multimodal(copy.deepcopy(list_data_dict), multimodal_cfg, 576)  # HARDCODED FOR NOW
        data_dict = preprocess(sources, self.tokenizer, multimodal_cfg)
        return data_dict['tokens'].tolist()

    def tokenize_batch(self, prompt, max_len, add_BOS):
        context_tokens = self.process_prompts(prompt)
        context_tokens, context_lengths = pad_batch(context_tokens, self.tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    def get_media_tensor(self, image_path):
        from PIL import Image

        image = Image.open(image_path).convert('RGB')

        if self.data_cfg.image_aspect_ratio == 'keep':
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 448, 224
            shortest_edge = int(min(max_len / aspect_ratio, min_len))
            image = self.processor.preprocess(
                image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge}
            )['pixel_values'][0]
        elif self.data_cfg.image_aspect_ratio == 'pad':

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in self.processor.image_mean))
            image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        model_cfg = self.model.cfg

        if model_cfg.precision == 16:
            media = image.type(torch.float16)
        elif model_cfg.precision == 32:
            media = image.type(torch.float32)
        else:
            media = image.type(torch.bfloat16)

        return media.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool = True,
        media=None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
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
        attention_mask_repeat = None
        if compute_attention_mask:
            attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])

        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())
        batch = [tokens2use, attention_mask_repeat, positions2use, media, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.encoder_seq_length]
        return batch, tensor_shape


class PromptLearningModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model, task_ids):
        super().__init__(model)
        self.task_ids = task_ids
        self.forward_model = self.model

    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
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
            compute_attention_mask=compute_attention_mask,
        )

    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length"""
        if maxlen > self.model.frozen_model.cfg.encoder_seq_length + 1:
            maxlen = self.model.frozen_model.cfg.encoder_seq_length + 1
        return maxlen

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool,
    ) -> Tuple[List[torch.Tensor], List[int]]:
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
        attention_mask_repeat = None
        if compute_attention_mask:
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
            tokens[:, :context_length][(tokens[:, :context_length] >= pseudo_token_ids_start)] = tokenizer.unk_id


def model_inference_strategy_dispatcher(model, **args):
    from nemo.collections.multimodal.models.neva.neva_model import MegatronNevaModel
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
        MegatronGPTPromptLearningModel,
    )
    from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
    from nemo.collections.nlp.modules.common.retro_inference_strategies import (
        RetroFileQAModelTextGenerationStrategy,
        RetroModelTextGenerationStrategy,
        RetroQAModelTextGenerationStrategy,
    )

    if isinstance(model, MegatronNevaModel):
        return NevaModelTextGenerationStrategy(model)
    if isinstance(model, MegatronGPTPromptLearningModel):
        return PromptLearningModelTextGenerationStrategy(model, **args)
    elif isinstance(model, MegatronGPTModel):
        return GPTModelTextGenerationStrategy(model)
    elif isinstance(model, MegatronRetrievalModel):
        strategy_name = args['strategy']
        del args['strategy']
        megatron_lm_compatible = model.model.megatron_lm_compatible
        args['megatron_lm_compatible'] = megatron_lm_compatible
        if strategy_name == 'RetroModelTextGenerationStrategy':
            return RetroModelTextGenerationStrategy(model, **args)
        elif strategy_name == 'RetroQAModelTextGenerationStrategy':
            return RetroQAModelTextGenerationStrategy(model, **args)
        elif strategy_name == 'RetroFileQAModelTextGenerationStrategy':
            return RetroFileQAModelTextGenerationStrategy(model, **args)
        else:
            raise ValueError(f'{strategy_name} is not supported for inference')
    else:
        raise ValueError(f'{model} is not supported for inference')

    # Should call GPTModel or Megatron Retrieval Model's forward method
