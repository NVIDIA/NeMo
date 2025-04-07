# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import transformers
from transformers import AutoImageProcessor
from transformers.image_utils import ImageInput, is_valid_image, load_image

from cosmos1.models.autoregressive.tokenizer.text_tokenizer import TextTokenizer
from cosmos1.utils import log

# Configuration for different vision-language models
IMAGE_CONFIGS = {
    "pixtral": {
        "patch_size": 16,
        "image_token": "[IMG]",
        "image_break_token": "[IMG_BREAK]",
        "image_end_token": "[IMG_END]",
    }
}

# Chat template for Pixtral-12B-Instruct
PIXTRAL_CHAT_TEMPLATE = '{%- if messages[0]["role"] == "system" %}\n    {%- set system_message = messages[0]["content"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message[\'role\'] == \'user\') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception(\'After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\') }}\n    {%- endif %}\n    {%- if message["role"] == "user" %}\n        {%- if loop.last and system_message is defined %}\n            {{- "[INST]" + system_message + "\n\n" }}\n        {%- else %}\n            {{- "[INST]" }}\n        {%- endif %}\n        {%- if message["content"] is not string %}\n            {%- for chunk in message["content"] %}\n                {%- if chunk["type"] == "text" %}\n                    {{- chunk["content"] }}\n                {%- elif chunk["type"] == "image" %}\n                    {{- "[IMG]" }}\n                {%- else %}\n                    {{- raise_exception("Unrecognized content type!") }}\n                {%- endif %}\n            {%- endfor %}\n        {%- else %}\n            {{- message["content"] }}\n        {%- endif %}\n        {{- "[/INST]" }}\n    {%- elif message["role"] == "assistant" %}\n        {{- message["content"] + eos_token}}\n    {%- else %}\n        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}\n    {%- endif %}\n{%- endfor %}'


# Copied from transformers.models.pixtral.processing_pixtral.is_url
def is_url(val) -> bool:
    """Check if the given value is a URL."""
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.pixtral.processing_pixtral.is_image_or_image_url
def is_image_or_image_url(elem):
    """Check if the given element is an image or an image URL."""
    return is_url(elem) or is_valid_image(elem)


def load_image_list(
    image_list: List[Union[str, "PIL.Image.Image"]], timeout: Optional[float] = None
) -> List["PIL.Image.Image"]:
    """
    Load a list of images.

    Args:
        image_list (List[Union[str, PIL.Image.Image]]): The list of images to load.
        timeout (Optional[float]): The timeout for loading the image.

    Returns:
        List[PIL.Image.Image]: The list of loaded images.
    """
    return [load_image(image, timeout=timeout) for image in image_list]


class ImageTextTokenizer(TextTokenizer):
    """
    Image-text tokenizer class that extends the text tokenizer to support vision tokens as well.
    """

    def __init__(
        self,
        model_family: str,
        is_instruct_model: bool,
        tokenizer_path: str,
        image_processor_path: str,
    ):
        """
        Initialize the ImageTextTokenizer.

        Args:
            model_family (str): The model family.
            is_instruct_model (bool): Whether the model is an instruct model.
            s3_credential_path (str): The path to the s3 credential file. Defaults to "credentials/pbss_dir.secret".

        Raises:
            AssertionError: If the model family is not supported or if the transformers version is incompatible.
        """
        super().__init__(
            model_family=model_family,
            is_instruct_model=is_instruct_model,
            local_path=tokenizer_path,
        )
        assert model_family in ["pixtral"], f"Unsupported model family: {model_family}"
        if model_family == "pixtral":
            # Need transformers>=4.45.0
            assert transformers.__version__ >= "4.45.0", "Pixtral requires transformers>=4.45.0"
            assert is_instruct_model, "Pixtral requires is_instruct_model=True"
            if not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None:
                setattr(self.tokenizer, "chat_template", PIXTRAL_CHAT_TEMPLATE)
                log.debug(f"Pixtral tokenizer chat template set to: {PIXTRAL_CHAT_TEMPLATE}")

        # Set up image-specific configurations
        image_config = IMAGE_CONFIGS[model_family]
        self.patch_size = image_config["patch_size"]
        self.image_token = image_config["image_token"]
        self.image_break_token = image_config["image_break_token"]
        self.image_end_token = image_config["image_end_token"]

        # Initialize the image processor
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_path)

    def encode(
        self,
        text: Union[str, List[str], List[int]],
        *,  # Enforce keyword-only arguments
        images: Optional[ImageInput] = None,
        image_kwargs: Optional[Dict[str, Any]] = None,
        **text_kwargs,
    ) -> List[int]:
        """
        Process the images and return the tokenized images and text.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded.
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared.
            image_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for image processing.
            **text_kwargs: Additional keyword arguments for text processing.

        Returns:
            A dictionary with the following fields:
            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **pixel_values** -- Pixel values to be fed to a model.

        Raises:
            ValueError: If the input images are in an invalid format.
        """

        output_dict, image_inputs = {}, {}
        if images is not None:
            # Preprocess images
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, list) and is_image_or_image_url(images[0]):
                images = [images]
            elif (
                not isinstance(images, list)
                and not isinstance(images[0], list)
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )

            # Load and process images
            images = [load_image_list(sample) for sample in images]
            image_kwargs = image_kwargs or {}
            image_inputs = self.image_processor(images, patch_size=self.patch_size, return_tensors="np", **image_kwargs)

            # Validate image inputs
            assert "pixel_values" in image_inputs, "pixel_values not found in image_inputs"
            assert "image_sizes" in image_inputs, "image_sizes not found in image_inputs"
            assert len(image_inputs.keys()) == 2, "Only one key is allowed in image_inputs, got {}".format(
                image_inputs.keys()
            )

            # Extract pixel values and image sizes
            pixel_values = image_inputs["pixel_values"][0]
            image_sizes = image_inputs["image_sizes"][0]
            unique_sizes = np.unique(image_sizes, axis=0)

            assert len(unique_sizes) == 1, "All images must have the same size, got {}".format(unique_sizes)

            # Convert pixel values to PyTorch tensor
            pixel_values = np.asarray(pixel_values)
            pixel_values = torch.from_numpy(pixel_values)
            output_dict["pixel_values"] = pixel_values
            output_dict["image_sizes"] = image_sizes

        # Expand image tokens in text
        if image_inputs.get("pixel_values") is not None:
            replace_strings = []
            # Calculate the number of tokens needed for each image and create a placeholder
            for image_size in image_sizes:
                height, width = image_size
                num_height_tokens = height // self.patch_size
                num_width_tokens = width // self.patch_size
                replace_tokens = [[self.image_token] * num_width_tokens + [self.image_break_token]] * num_height_tokens
                # Flatten list
                replace_tokens = [item for sublist in replace_tokens for item in sublist]
                replace_tokens[-1] = self.image_end_token
                replace_str = "".join(replace_tokens)
                replace_strings.append(replace_str)
                text = text.replace(self.image_token, "<placeholder>", 1)

            # Replace placeholders with actual image token sequences
            while "<placeholder>" in text:
                replace_str = replace_strings.pop(0)
                text = text.replace("<placeholder>", replace_str, 1)

        # Encode the text
        text_inputs = super(ImageTextTokenizer, self).encode(text, **text_kwargs)

        output_dict["input_ids"] = text_inputs
        return output_dict

    def apply_chat_template(
        self,
        conversation: List[Dict[str, Any]] | List[List[Dict[str, Any]]],
        *,
        images: Optional[ImageInput] = None,
        image_kwargs: Optional[Dict[str, Any]] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_dict: bool = True,
        return_assistant_tokens_mask: bool = False,
        generation_prefix: str = "",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Apply the chat template to the conversation.

        Args:
            conversation (List[Dict[str, Any]] | List[List[Dict[str, Any]]]): The conversation to process.
            images (Optional[ImageInput]): Images to include in the conversation.
            image_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for image processing.
            add_generation_prompt (bool): Whether to add a generation prompt.
            tokenize (bool): Whether to tokenize the output.
            padding (bool): Whether to pad the output.
            truncation (bool): Whether to truncate the output.
            max_length (Optional[int]): Maximum length of the output.
            return_tensors (Optional[str]): The type of tensors to return.
            return_dict (bool): Whether to return a dictionary.
            return_assistant_tokens_mask (bool): Whether to return the assistant tokens mask.
            generation_prefix (str): Prefix to add before asking model to generate. Helpful to guide the generation. Defaults to "".
            tokenizer_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the tokenizer.
            **kwargs: Additional keyword arguments.

        Returns:
            The processed conversation with applied chat template.

        Raises:
            AssertionError: If return_dict is False or if the conversation format is invalid.
        """
        assert return_dict, "return_dict must be True for ImageTextTokenizer"
        assert isinstance(conversation, list), "conversation must be a list"
        if isinstance(conversation[0], list):
            assert len(conversation) == 1, "Only support single-conversation input, got {}".format(conversation)
            conversation = conversation[0]

        # Extract images from the conversation if not provided
        if images is None:
            images = []
            for msg in conversation:
                if msg.get("images", None) is not None:
                    images = images + (msg["images"])
            images = load_image_list(images)
        # In case the input does not have images, will ignore
        # Useful in feeding VLM inputs with and without images
        if isinstance(images, list) and len(images) == 0:
            images = None

        # Apply the chat template to the text
        text = super().apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=False,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            generation_prefix=generation_prefix,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        # Encode the text and images
        output = self.encode(
            text,
            images=images,
            image_kwargs=image_kwargs,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )
        return output

    @property
    def model_input_names(self):
        """
        Get the combined model input names from both the text tokenizer and image processor.

        Returns:
            List[str]: A list of unique input names.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
