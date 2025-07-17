# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# list of extra tokens

def generate_extra_grounding_tokens(tokenizer, num_image_think_tokens=16, num_bbox_tokens=16):
    ''' 
    Given tokenizer, generate extra tokens for image thinking and bbox tokens
    Currently, the extra tokens added are:
    - <|img_think_end|>   # to delineate the think tokens and the rest
    - <|img_think_0|>
    - <|img_think_1|>
    - ...
    - <|bbox_0|>
    - <|bbox_1|>
    - ...
    - <|count|>
    '''

    img_think_end_token = "<|img_think_end|>"
    extra_tokens = [img_think_end_token]
    for i in range(num_image_think_tokens):
        extra_tokens.append(f"<|img_think_{i}|>")
    for i in range(num_bbox_tokens):
        extra_tokens.append(f"<|bbox_{i}|>")
    extra_tokens.append("<|count|>")

    # capture extra metadata to be used in the model
    metadata = {
        'num_image_think_tokens': num_image_think_tokens,
        'num_bbox_tokens': num_bbox_tokens,
    }

    # add them to the tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": extra_tokens})
    # get ids
    extra_tokens_ids = tokenizer.encode("".join(extra_tokens))
    assert len(extra_tokens_ids) == len(extra_tokens), "Number of tokens and ids do not match"
    # add them to the model
    return tokenizer, extra_tokens, extra_tokens_ids, metadata
