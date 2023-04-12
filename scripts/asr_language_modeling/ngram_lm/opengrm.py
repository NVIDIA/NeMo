# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#

"""
Utility methods to be used to merge arpa N-gram language models (LMs), 
culculate perplexity of resulted LM, and make binary KenLM from it.

Minimun usage example to merge two N-gram language models with weights:
alpha * ngram_a + beta * ngram_b = 2 * ngram_a + 1 * ngram_b

python3 opengrm.py  --kenlm_bin_path /workspace/nemo/decoders/kenlm/build/bin/build_binary \
                    --arpa_a /path/ngram_a.kenlm.tmp.arpa \
                    --alpha 2 \
                    --arpa_b /path/ngram_b.kenlm.tmp.arpa \
                    --beta 1 \
                    --out_path /path/out \


Merge two N-gram language models and calculate its perplexity with test_file.
python3 opengrm.py  --kenlm_bin_path /workspace/nemo/decoders/kenlm/build/bin/build_binary \
                    --arpa_a /path/ngram_a.kenlm.tmp.arpa \
                    --alpha 0.5 \
                    --arpa_b /path/ngram_b.kenlm.tmp.arpa \
                    --beta 0.5 \
                    --out_path /path/out \
                    --tokenizer_model_file /path/to/model_tokenizer.nemo \
                    --test_file /path/to/test_manifest.json \
                    --force
"""

import argparse
import os
import subprocess
import sys

import kenlm_utils
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.ctc_beam_decoding import DEFAULT_TOKEN_OFFSET
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.utils import logging


def ngrammerge(arpa_a, alpha, arpa_b, beta, arpa_c, force):
    mod_a = arpa_a + ".mod"
    mod_b = arpa_b + ".mod"
    mod_c = arpa_c + ".mod"
    if os.path.isfile(mod_c) and not force:
        logging.info("File " + mod_c + " exists. Skipping.")
    else:
        sh_args = [
            "ngrammerge",
            "--alpha=" + str(alpha),
            "--beta=" + str(beta),
            "--normalize",
            # "--use_smoothing",
            mod_a,
            mod_b,
            mod_c,
        ]
        logging.info(
            "\n"
            + str(subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,))
            + "\n",
        )
    return mod_c


def arpa2mod(arpa_path, force):
    mod_path = arpa_path + ".mod"
    if os.path.isfile(mod_path) and not force:
        return "File " + mod_path + " exists. Skipping."
    else:
        sh_args = [
            "ngramread",
            "--ARPA",
            arpa_path,
            mod_path,
        ]
        return subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,)


def merge(arpa_a, alpha, arpa_b, beta, out_path, force):
    logging.info("\n" + str(arpa2mod(arpa_a, force)) + "\n")

    logging.info("\n" + str(arpa2mod(arpa_b, force)) + "\n")
    arpa_c = os.path.join(out_path, f"{os.path.split(arpa_a)[1]}-{alpha}-{os.path.split(arpa_b)[1]}-{beta}.arpa",)
    mod_c = ngrammerge(arpa_a, alpha, arpa_b, beta, arpa_c, force)
    return mod_c, arpa_c


def make_symbol_list(tokenizer_model_file, symbols, force):
    if os.path.isfile(symbols) and not force:
        logging.info("File " + symbols + " exists. Skipping.")
    else:
        if tokenizer_model_file.endswith('.model'):
            tokenizer_nemo = SentencePieceTokenizer(tokenizer_model_file)
            vocab_size = tokenizer_nemo.vocab_size
        elif tokenizer_model_file.endswith('.nemo'):
            asr_model = nemo_asr.models.ASRModel.restore_from(tokenizer_model_file, map_location=torch.device('cpu'))
            vocab_size = len(asr_model.decoder.vocabulary)
        else:
            logging.warning(
                "tokenizer_model_file does not end with .model or .nemo, therefore trying to load a pretrained model with this name."
            )
            asr_model = nemo_asr.models.ASRModel.from_pretrained(
                tokenizer_model_file, map_location=torch.device('cpu')
            )
            vocab_size = len(asr_model.decoder.vocabulary)

        vocab = [chr(idx + DEFAULT_TOKEN_OFFSET) for idx in range(vocab_size)]
        with open(symbols, "w", encoding="utf-8") as f:
            for i, v in enumerate(vocab):
                f.write(v + " " + str(i) + "\n")


def farcompile(
    symbols, text_file, test_far, tokenizer_model_file, do_lowercase, rm_punctuation, separate_punctuation, force,
):
    if os.path.isfile(test_far) and not force:
        logging.info("File " + test_far + " exists. Skipping.")
        return
    else:
        sh_args = [
            "farcompilestrings",
            "--generate_keys=10",
            "--fst_type=compact",
            "--symbols=" + symbols,
            "--keep_symbols",
            ">",
            test_far,
        ]

        tokenizer, encoding_level, is_aggregate_tokenizer = kenlm_utils.setup_tokenizer(tokenizer_model_file)

        ps = subprocess.Popen(
            " ".join(sh_args), shell=True, stdin=subprocess.PIPE, stdout=sys.stdout, stderr=sys.stderr,
        )

        kenlm_utils.iter_files(
            ps.stdin,
            [text_file],
            tokenizer,
            encoding_level,
            is_aggregate_tokenizer,
            do_lowercase,
            rm_punctuation,
            separate_punctuation,
            verbose=0,
        )
        stdout, stderr = ps.communicate()

        exit_code = ps.returncode

        command = " ".join(sh_args)
        assert (
            exit_code == 0
        ), f"Exit_code must be 0.\n bash command: {command} \n stdout: {stdout} \n stderr: {stderr}"
        return stdout, stderr


def perplexity(ngram_mod, test_far):
    sh_args = [
        "ngramperplexity",
        "--v=1",
        ngram_mod,
        test_far,
    ]
    ps = subprocess.Popen(sh_args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = ps.communicate()
    exit_code = ps.wait()
    command = " ".join(sh_args)
    assert exit_code == 0, f"Exit_code must be 0.\n bash command: {command} \n stdout: {stdout} \n stderr: {stderr}"
    perplexity_out = "\n".join(stdout.split("\n")[-6:-1])
    return perplexity_out


def make_arpa(ngram_mod, ngram_arpa, force):
    if os.path.isfile(ngram_arpa) and not force:
        logging.info("File " + ngram_arpa + " exists. Skipping.")
        return
    else:
        sh_args = [
            "ngramprint",
            "--ARPA",
            ngram_mod,
            ngram_arpa,
        ]
        return subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,)


def make_kenlm(kenlm_bin_path, ngram_arpa, force):
    ngram_kenlm = ngram_arpa + ".kenlm"
    if os.path.isfile(ngram_kenlm) and not force:
        logging.info("File " + ngram_kenlm + " exists. Skipping.")
        return
    else:
        sh_args = [kenlm_bin_path, "trie", "-i", ngram_arpa, ngram_kenlm]
        return subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,)


def test_perplexity(mod_c, symbols, test_txt, tokenizer_model_file, tmp_path, force):
    test_far = os.path.join(tmp_path, os.path.split(test_txt)[1] + ".far")
    farcompile(symbols, test_txt, test_far, tokenizer_model_file, False, False, False, force)
    res_p = perplexity(mod_c, test_far)
    return res_p


def main(kenlm_bin_path, arpa_a, alpha, arpa_b, beta, out_path, test_file, symbols, tokenizer_model_file, force):

    mod_c, arpa_c = merge(arpa_a, alpha, arpa_b, beta, out_path, force)

    if test_file and tokenizer_model_file:
        if not symbols:
            symbols = os.path.join(out_path, os.path.split(tokenizer_model_file)[1] + ".syms")
            make_symbol_list(tokenizer_model_file, symbols, force)
        test_p = test_perplexity(mod_c, symbols, test_file, tokenizer_model_file, out_path, force)
        logging.info("Perplexity summary:" + test_p)

    logging.info("Making ARPA and Kenlm model " + arpa_c)
    out = make_arpa(mod_c, arpa_c, force)
    if out:
        logging.info("\n" + str(out) + "\n")

    out = make_kenlm(kenlm_bin_path, arpa_c, force)
    if out:
        logging.info("\n" + str(out) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="Avg pytorch weights")
    parser.add_argument(
        "--kenlm_bin_path", required=True, type=str, help="The path to the bin folder of KenLM",
    )  # Use /workspace/nemo/decoders/kenlm/build/bin/build_binary if installed it with scripts/installers/install_beamsearch_decoders.sh
    parser.add_argument("--arpa_a", required=True, type=str, help="Path to the arpa_a")
    parser.add_argument("--alpha", required=True, type=float, help="Weight of arpa_a")
    parser.add_argument("--arpa_b", required=True, type=str, help="Path to the arpa_b")
    parser.add_argument("--beta", required=True, type=float, help="Weight of arpa_b")
    parser.add_argument(
        "--out_path", required=True, type=str, help="Path to write tmp and resulted files",
    )
    parser.add_argument(
        "--test_file",
        required=False,
        type=str,
        default=None,
        help="Path to test file to count perplexity if provided.",
    )
    parser.add_argument(
        "--symbols",
        required=False,
        type=str,
        default=None,
        help="Path to symbols file. Could be calculated if it is not provided. Used as: --symbols /path/to/earnest.syms",
    )
    parser.add_argument(
        "--tokenizer_model_file",
        required=False,
        type=str,
        default=None,
        help="The path to '.model' file of the SentencePiece tokenizer, or '.nemo' file of the ASR model, or name of a pretrained NeMo model",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Whether to compile and rewrite all files")
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(_parse_args()))
