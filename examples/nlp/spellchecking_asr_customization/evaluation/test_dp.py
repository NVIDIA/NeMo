# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from argparse import ArgumentParser

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    get_alignment_by_dp,
    load_ngram_mappings_for_dp,
)

parser = ArgumentParser(description="Get shortest path by n-gram mappings")
parser.add_argument("--ngram_mappings", required=True, type=str, help="Path to ngram mappings file")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()


joint_vocab, src_vocab, dst_vocab, max_len = load_ngram_mappings_for_dp(args.ngram_mappings)

hyp_phrase = "i n a c c e s s i b l e"
ref_phrase = "a c c e s s i b l e"
path = get_alignment_by_dp(hyp_phrase, ref_phrase, joint_vocab, src_vocab, dst_vocab, max_len)
for hyp_ngram, ref_ngram, score, sum_score, joint_freq, src_freq, dst_freq in path:
    print(
        "\t",
        "hyp=",
        hyp_ngram,
        "; ref=",
        ref_ngram,
        "; score=",
        score,
        "; sum_score=",
        sum_score,
        joint_freq,
        src_freq,
        dst_freq,
    )
