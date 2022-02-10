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
import signal
import os
import sys
import numpy as np

import inference_lib.text_generation as text_generation


def main():
    parser = argparse.ArgumentParser()

    text_generation.args_parser(parser)

    parser.add_argument(
        "-j", "--json-log", type=str, required=False, default=None, help="JSON log file"
    )
    parser.add_argument(
        "-c",
        "--customer",
        type=str,
        required=False,
        default="Customer question:",
        help="Text used for input prompt",
    )
    parser.add_argument(
        "-s",
        "--support",
        type=str,
        required=False,
        default="Helpdesk answer:",
        help="Text used for output prompt",
    )

    FLAGS, encoder, client_util = text_generation.args_validate(parser)

    if FLAGS.json_log is not None:
        if os.path.exists(FLAGS.json_log):
            print(f"WARNING: JSON log file {FLAGS.json_log} already exists")
            sys.exit(-1)

    def key_handler(signal, frame):
        print("Keyboard handler detected with signal ".format(signal))
        if len(input_output) > 0:
            if FLAGS.json_log is not None:
                print("Saving JSON log")
                import json

                with open(FLAGS.json_log, "w") as file:
                    json.dump(input_output, file, indent=4)
        exit(0)

    signal.signal(signal.SIGINT, key_handler)

    # Encode query string to get tokens
    print(FLAGS.customer, end="")
    question = input()
    question_with_marks = f'{FLAGS.customer} "{question}" {FLAGS.support} '
    encoded_text = encoder.encode(question_with_marks)

    input_output = {"queries": []}
    query = {
        "input": question_with_marks,
        "input_len": len(encoded_text),
        "output_len": FLAGS.output_len,
    }

    produced_tokens = 0

    while True:

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

        output_tokens_without_input = output_tokens_only[input_len:]
        produced_tokens += len(output_tokens_without_input)

        decoded = encoder.decode(output_tokens_without_input)
        query["output"] = decoded
        if FLAGS.json_log is not None:
            input_output["queries"].append(query)

        decoded_strip = decoded.strip()

        import re

        quoted = re.findall(r'"(.*?)"', decoded)
        if len(quoted) < 1:
            decoded_only_answer = "I don't understand. Can you rephrase your request?"
        else:
            decoded_only_answer = quoted[0]
        print(FLAGS.support, decoded_only_answer)

        print(FLAGS.customer, "(END to FINISH): ", end="")
        question = input()
        if question == "END":
            break
        question_with_marks = (
            FLAGS.support
            + ' "'
            + decoded_only_answer
            + '" '
            + FLAGS.customer
            + ' "'
            + question
            + '" '
            + FLAGS.support
            + " "
        )
        encoded_text = encoder.encode(question_with_marks)
        query = {
            "input": question_with_marks,
            "input_len": len(encoded_text),
            "output_len": FLAGS.output_len,
        }

    if len(input_output) > 0:
        if FLAGS.json_log is not None:
            import json

            with open(FLAGS.json_log, "w") as file:
                json.dump(input_output, file, indent=4)


if __name__ == "__main__":
    main()
