#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import argparse
import numpy as np

import inference_lib.text_generation as text_generation


def main():
    parser = argparse.ArgumentParser()

    text_generation.args_parser(parser)

    parser.add_argument(
        "-q", "--query", type=str, required=True, help="Query text send to model"
    )

    FLAGS, encoder, client_util = text_generation.args_validate(parser)

    # Encode query string to get tokens
    encoded_text = encoder.encode(FLAGS.query)

    # Prepare input tensors
    input_len = len(encoded_text)
    output_len = FLAGS.output_len

    input_tokens_tensor = np.array([[encoded_text]]).astype(np.uint32)
    input_len_tensor = np.array([[input_len]]).astype(np.uint32)
    output_len_tensor = np.array([[output_len]]).astype(np.uint32)

    # Send request to Triton
    output_ids = text_generation.send_requests(
        FLAGS.url,
        FLAGS.protocol,
        input_tokens_tensor,
        input_len_tensor,
        output_len_tensor,
        FLAGS.verbose,
        client_util,
        FLAGS.model_name,
    )

    # Decode output
    output_tokens = output_ids.squeeze()
    output_tokens_list = list([token for token in output_tokens])
    if FLAGS.end_id in output_tokens_list:
        end_id_pos = output_tokens_list.index(FLAGS.end_id)
        output_tokens_only = output_tokens_list[:end_id_pos]
    else:
        output_tokens_only = output_tokens_list

    decoded = encoder.decode(output_tokens_only[input_len:])
    print(decoded)


if __name__ == "__main__":
    main()
