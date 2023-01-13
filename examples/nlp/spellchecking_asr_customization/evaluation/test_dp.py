import json
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from nemo.collections.nlp.data.spellchecking_asr_customization.utils import load_ngram_mappings_for_dp, get_alignment_by_dp

parser = ArgumentParser(
    description="Analyze custom phrases recognition after ASR"
)
parser.add_argument("--ngram_mappings", required=True, type=str, help="Path to ngram mappings file")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()


joint_vocab, src_vocab, dst_vocab, max_len = load_ngram_mappings_for_dp(args.ngram_mappings)

hyp_phrase = "i n c e l i d u s"
ref_phrase = "e n c e l a d u s"
get_alignment_by_dp(hyp_phrase, ref_phrase, joint_vocab, src_vocab, dst_vocab, max_len)
