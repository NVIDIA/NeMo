# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from unittest import TestCase

from .chat_format import ChatFormat
from .datatypes import SystemMessage, UserMessage
from .tokenizer import Tokenizer


# TOKENIZER_PATH=<tokenizer_path> python -m unittest models/llama3/api/test_tokenizer.py


class TokenizerTests(TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(os.environ["TOKENIZER_PATH"])
        self.format = ChatFormat(self.tokenizer)

    def test_special_tokens(self):
        self.assertEqual(
            self.tokenizer.special_tokens["<|begin_of_text|>"],
            128000,
        )

    def test_encode(self):
        self.assertEqual(
            self.tokenizer.encode("This is a test sentence.", bos=True, eos=True),
            [128000, 2028, 374, 264, 1296, 11914, 13, 128001],
        )

    def test_decode(self):
        self.assertEqual(
            self.tokenizer.decode(
                [128000, 2028, 374, 264, 1296, 11914, 13, 128001],
            ),
            "<|begin_of_text|>This is a test sentence.<|end_of_text|>",
        )

    def test_encode_message(self):
        message = UserMessage(
            content="This is a test sentence.",
        )
        self.assertEqual(
            self.format.encode_message(message),
            [
                128006,  # <|start_header_id|>
                882,  # "user"
                128007,  # <|end_of_header|>
                271,  # "\n\n"
                2028,
                374,
                264,
                1296,
                11914,
                13,  # This is a test sentence.
                128009,  # <|eot_id|>
            ],
        )

    def test_encode_dialog(self):
        messages = [
            SystemMessage(
                content="This is a test sentence.",
            ),
            UserMessage(
                content="This is a response.",
            ),
        ]
        model_input = self.format.encode_dialog_prompt(messages)
        self.assertEqual(
            model_input.tokens,
            [
                128000,  # <|begin_of_text|>
                128006,  # <|start_header_id|>
                9125,  # "system"
                128007,  # <|end_of_header|>
                271,  # "\n\n"
                2028,
                374,
                264,
                1296,
                11914,
                13,  # "This is a test sentence."
                128009,  # <|eot_id|>
                128006,  # <|start_header_id|>
                882,  # "user"
                128007,  # <|end_of_header|>
                271,  # "\n\n"
                2028,
                374,
                264,
                2077,
                13,  # "This is a response.",
                128009,  # <|eot_id|>
                128006,  # <|start_header_id|>
                78191,  # "assistant"
                128007,  # <|end_of_header|>
                271,  # "\n\n"
            ],
        )
