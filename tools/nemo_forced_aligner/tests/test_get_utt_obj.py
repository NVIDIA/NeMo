# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from utils.data_prep import get_utt_obj, Token, Word, Segment, Utterance

from dataclasses import asdict

from nemo.collections.asr.models import ASRModel

import prettyprinter
from prettyprinter import register_pretty, pretty_call


def get_utt_obj_pp_string(utt_obj):
    @register_pretty(Word)
    def pretty_utterance(value, ctx):
        return pretty_call(
            ctx,
            Word,
            text=value.text,
            s_start=value.s_start,
            s_end=value.s_end,
            t_start=value.t_start,
            t_end=value.t_end,
            tokens=value.tokens,
        )

    @register_pretty(Segment)
    def pretty_utterance(value, ctx):
        return pretty_call(
            ctx,
            Segment,
            text=value.text,
            s_start=value.s_start,
            s_end=value.s_end,
            t_start=value.t_start,
            t_end=value.t_end,
            words_and_tokens=value.words_and_tokens,
        )

    @register_pretty(Utterance)
    def pretty_utterance(value, ctx):
        return pretty_call(
            ctx,
            Utterance,
            text=value.text,
            token_ids_with_blanks=value.token_ids_with_blanks,
            S=value.S,
            T=value.T,
            segments_and_tokens=value.segments_and_tokens,
        )

    return prettyprinter.pformat(utt_obj)


EN_TEXT = "hi world | hey"

EN_CN_EXPECTED_UTTERANCE = Utterance(
    text='hi world | hey',
    token_ids_with_blanks=[1024, 317, 1024, 472, 1024, 25, 1024, 20, 1024],
    S=9,
    T=None,
    segments_and_tokens=[
        Token(text='<b>', text_cased='<b>', s_start=0, s_end=0, t_start=None, t_end=None),
        Segment(
            text='hi world',
            s_start=1,
            s_end=3,
            t_start=None,
            t_end=None,
            words_and_tokens=[
                Word(
                    text='hi',
                    s_start=1,
                    s_end=1,
                    t_start=None,
                    t_end=None,
                    tokens=[Token(text='▁hi', text_cased='▁hi', s_start=1, s_end=1, t_start=None, t_end=None)],
                ),
                Token(text='<b>', text_cased='<b>', s_start=2, s_end=2, t_start=None, t_end=None),
                Word(
                    text='world',
                    s_start=3,
                    s_end=3,
                    t_start=None,
                    t_end=None,
                    tokens=[Token(text='▁world', text_cased='▁world', s_start=3, s_end=3, t_start=None, t_end=None)],
                ),
            ],
        ),
        Token(text='<b>', text_cased='<b>', s_start=4, s_end=4, t_start=None, t_end=None),
        Segment(
            text='hey',
            s_start=5,
            s_end=7,
            t_start=None,
            t_end=None,
            words_and_tokens=[
                Word(
                    text='hey',
                    s_start=5,
                    s_end=7,
                    t_start=None,
                    t_end=None,
                    tokens=[
                        Token(text='▁he', text_cased='▁he', s_start=5, s_end=5, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=6, s_end=6, t_start=None, t_end=None),
                        Token(text='y', text_cased='y', s_start=7, s_end=7, t_start=None, t_end=None),
                    ],
                )
            ],
        ),
        Token(text='<b>', text_cased='<b>', s_start=8, s_end=8, t_start=None, t_end=None),
    ],
)

EN_QN_EXPECTED_UTTERANCE = Utterance(
    text='hi world | hey',
    token_ids_with_blanks=[
        28,
        8,
        28,
        9,
        28,
        0,
        28,
        23,
        28,
        15,
        28,
        18,
        28,
        12,
        28,
        4,
        28,
        0,
        28,
        8,
        28,
        5,
        28,
        25,
        28,
    ],
    S=25,
    T=None,
    segments_and_tokens=[
        Token(text='<b>', text_cased='<b>', s_start=0, s_end=0, t_start=None, t_end=None),
        Segment(
            text="hi world",
            s_start=1,
            s_end=15,
            t_start=None,
            t_end=None,
            words_and_tokens=[
                Word(
                    text="hi",
                    s_start=1,
                    s_end=3,
                    t_start=None,
                    t_end=None,
                    tokens=[
                        Token(text='h', text_cased='h', s_start=1, s_end=1, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=2, s_end=2, t_start=None, t_end=None),
                        Token(text='i', text_cased='i', s_start=3, s_end=3, t_start=None, t_end=None),
                    ],
                ),
                Token(text='<b>', text_cased='<b>', s_start=4, s_end=4, t_start=None, t_end=None),
                Token(text='<space>', text_cased='<space>', s_start=5, s_end=5, t_start=None, t_end=None),
                Token(text='<b>', text_cased='<b>', s_start=6, s_end=6, t_start=None, t_end=None),
                Word(
                    text="world",
                    s_start=7,
                    s_end=15,
                    t_start=None,
                    t_end=None,
                    tokens=[
                        Token(text='w', text_cased='w', s_start=7, s_end=7, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=8, s_end=8, t_start=None, t_end=None),
                        Token(text='o', text_cased='o', s_start=9, s_end=9, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=10, s_end=10, t_start=None, t_end=None),
                        Token(text='r', text_cased='r', s_start=11, s_end=11, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=12, s_end=12, t_start=None, t_end=None),
                        Token(text='l', text_cased='l', s_start=13, s_end=13, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=14, s_end=14, t_start=None, t_end=None),
                        Token(text='d', text_cased='d', s_start=15, s_end=15, t_start=None, t_end=None),
                    ],
                ),
            ],
        ),
        Token(text='<b>', text_cased='<b>', s_start=16, s_end=16, t_start=None, t_end=None),
        Token(text='<space>', text_cased='<space>', s_start=17, s_end=17, t_start=None, t_end=None),
        Token(text='<b>', text_cased='<b>', s_start=18, s_end=18, t_start=None, t_end=None),
        Segment(
            text="hey",
            s_start=19,
            s_end=23,
            t_start=None,
            t_end=None,
            words_and_tokens=[
                Word(
                    text="hey",
                    s_start=19,
                    s_end=23,
                    t_start=None,
                    t_end=None,
                    tokens=[
                        Token(text='h', text_cased='h', s_start=19, s_end=19, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=20, s_end=20, t_start=None, t_end=None),
                        Token(text='e', text_cased='e', s_start=21, s_end=21, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=22, s_end=22, t_start=None, t_end=None),
                        Token(text='y', text_cased='y', s_start=23, s_end=23, t_start=None, t_end=None),
                    ],
                )
            ],
        ),
        Token(text='<b>', text_cased='<b>', s_start=24, s_end=24, t_start=None, t_end=None),
    ],
)


ZH_TEXT = "人工 智能|技术"

ZH_CN_EXPECTED_UTTERANCE = Utterance(
    text='人工 智能|技术',
    token_ids_with_blanks=[
        5206,
        125,
        5206,
        1329,
        5206,
        0,
        5206,
        2029,
        5206,
        3668,
        5206,
        0,
        5206,
        1695,
        5206,
        2075,
        5206,
    ],
    S=17,
    T=None,
    segments_and_tokens=[
        Token(text='<b>', text_cased='<b>', s_start=0, s_end=0, t_start=None, t_end=None),
        Segment(
            text='人工 智能',
            s_start=1,
            s_end=9,
            t_start=None,
            t_end=None,
            words_and_tokens=[
                Word(
                    text='人工',
                    s_start=1,
                    s_end=3,
                    t_start=None,
                    t_end=None,
                    tokens=[
                        Token(text='人', text_cased='人', s_start=1, s_end=1, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=2, s_end=2, t_start=None, t_end=None),
                        Token(text='工', text_cased='工', s_start=3, s_end=3, t_start=None, t_end=None),
                    ],
                ),
                Token(text='<b>', text_cased='<b>', s_start=4, s_end=4, t_start=None, t_end=None),
                Token(text='<space>', text_cased='<space>', s_start=5, s_end=5, t_start=None, t_end=None),
                Token(text='<b>', text_cased='<b>', s_start=6, s_end=6, t_start=None, t_end=None),
                Word(
                    text='智能',
                    s_start=7,
                    s_end=9,
                    t_start=None,
                    t_end=None,
                    tokens=[
                        Token(text='智', text_cased='智', s_start=7, s_end=7, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=8, s_end=8, t_start=None, t_end=None),
                        Token(text='能', text_cased='能', s_start=9, s_end=9, t_start=None, t_end=None),
                    ],
                ),
            ],
        ),
        Token(text='<b>', text_cased='<b>', s_start=10, s_end=10, t_start=None, t_end=None),
        Token(text='<space>', text_cased='<space>', s_start=11, s_end=11, t_start=None, t_end=None),
        Token(text='<b>', text_cased='<b>', s_start=12, s_end=12, t_start=None, t_end=None),
        Segment(
            text='技术',
            s_start=13,
            s_end=15,
            t_start=None,
            t_end=None,
            words_and_tokens=[
                Word(
                    text='技术',
                    s_start=13,
                    s_end=15,
                    t_start=None,
                    t_end=None,
                    tokens=[
                        Token(text='技', text_cased='技', s_start=13, s_end=13, t_start=None, t_end=None),
                        Token(text='<b>', text_cased='<b>', s_start=14, s_end=14, t_start=None, t_end=None),
                        Token(text='术', text_cased='术', s_start=15, s_end=15, t_start=None, t_end=None),
                    ],
                )
            ],
        ),
        Token(text='<b>', text_cased='<b>', s_start=16, s_end=16, t_start=None, t_end=None),
    ],
)


@pytest.mark.parametrize(
    "text,model_pretrained_name,separator,expected_utterance",
    [
        (EN_TEXT, "stt_en_citrinet_256_gamma_0_25", "|", EN_CN_EXPECTED_UTTERANCE),
        (EN_TEXT, "stt_en_quartznet15x5", "|", EN_QN_EXPECTED_UTTERANCE),
        (ZH_TEXT, "stt_zh_citrinet_512", "|", ZH_CN_EXPECTED_UTTERANCE),
    ],
)
def test_token_info(text, model_pretrained_name, separator, expected_utterance):
    model = ASRModel.from_pretrained(model_pretrained_name)
    utt_obj = get_utt_obj(text, model, separator)
    print(f"expected utterance object: {get_utt_obj_pp_string(expected_utterance)}\n")
    print(f"output utterance object in test: {get_utt_obj_pp_string(utt_obj)}\n")

    assert utt_obj == expected_utterance
