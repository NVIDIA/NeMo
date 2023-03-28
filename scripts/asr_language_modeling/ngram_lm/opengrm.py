import argparse
import os
import subprocess
import sys

from nemo.collections.asr.parts.submodules.ctc_beam_decoding import DEFAULT_TOKEN_OFFSET

NGRAM_BIN_PATH = "/root/miniconda3/bin/"
KENLM_BIN_PATH = "/workspace/nemo/decoders/kenlm/build/bin/build_binary"

# Example
# python3 opengrm.py --arpa_a /path/ngram_a.kenlm.tmp.arpa \
#                     --alpha 2 \
#                     --arpa_b /path/ngram_b.kenlm.tmp.arpa \
#                     --beta 1 \
#                     --out_path /path/out \


def ngrammerge(arpa_a, alpha, arpa_b, beta, arpa_c):
    mod_a = arpa_a + ".mod"
    mod_b = arpa_b + ".mod"
    mod_c = arpa_c + ".mod"
    if os.path.isfile(mod_c):
        print("File", mod_c, "exists. Skipping.")
    else:
        sh_args = [
            os.path.join(NGRAM_BIN_PATH, "ngrammerge"),
            "--alpha=" + str(alpha),
            "--beta=" + str(beta),
            "--normalize",
            # "--use_smoothing",
            mod_a,
            mod_b,
            mod_c,
        ]
        print(
            "\n",
            subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,),
            "\n",
        )
    return mod_c


def arpa2mod(arpa_path, force):
    mod_path = arpa_path + ".mod"
    if os.path.isfile(mod_path) and not force:
        return "File " + mod_path + " exists. Skipping."
    else:
        sh_args = [
            os.path.join(NGRAM_BIN_PATH, "ngramread"),
            "--ARPA",
            arpa_path,
            mod_path,
        ]
        return subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,)


def merge(arpa_a, alpha, arpa_b, beta, out_path, force):
    print("\n", arpa2mod(arpa_a, force), "\n")

    print("\n", arpa2mod(arpa_b, force), "\n")
    arpa_c = os.path.join(out_path, f"{os.path.split(arpa_a)[1]}-{alpha}-{os.path.split(arpa_b)[1]}-{beta}.arpa",)
    mod_c = ngrammerge(arpa_a, alpha, arpa_b, beta, arpa_c)
    return mod_c, arpa_c


def make_symbol_list(asr_model, symbols, force):
    if os.path.isfile(symbols) and not force:
        print("File", symbols, "exists. Skipping.")
    else:
        vocab = [chr(idx + DEFAULT_TOKEN_OFFSET) for idx in range(len(asr_model.decoder.vocabulary))]
        with open(symbols, "w", encoding="utf-8") as f:
            for i, v in enumerate(vocab):
                print(v, i)
                f.write(v + " " + str(i) + "\n")


def farcompile(
    symbols, text_file, test_far, nemo_model_file, clean_text, do_lowercase, rm_punctuation,
):
    file_path = os.path.split(os.path.realpath(__file__))[0]
    if os.path.isfile(test_far):
        print("File", test_far, "exists. Skipping.")
        return
    else:
        sh_args = [
            os.path.join(NGRAM_BIN_PATH, "farcompilestrings"),
            "--generate_keys=10",
            "--fst_type=compact",
            "--symbols=" + symbols,
            "--keep_symbols",
            ">",
            test_far,
        ]

        if os.path.split(text_file)[0][-5:] == "cache":
            first_process_args = ["cat", text_file]
        else:
            first_process_args = [
                "python3",
                os.path.join(file_path, "encode_text.py"),
                "--nemo_model_file",
                nemo_model_file,
                "--train_path",
                text_file,
            ]
            if clean_text:
                first_process_args.append("--clean_text")
            if do_lowercase:
                first_process_args.append("--do_lowercase")
            if rm_punctuation:
                first_process_args.append("--rm_punctuation")

        sh_args = first_process_args + ["|"] + sh_args

        ps = subprocess.Popen(" ".join(sh_args), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,)
        return ps.communicate()[0]


def perplexity(ngram_mod, test_far):
    sh_args = [
        os.path.join(NGRAM_BIN_PATH, "ngramperplexity"),
        "--v=1",
        ngram_mod,
        test_far,
        "--OOV_symbol=d",
    ]
    ps = subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)
    return ps


def make_arpa(ngram_mod, ngram_arpa, force):
    if os.path.isfile(ngram_arpa) and not force:
        print("File", ngram_arpa, "exists. Skipping.")
        return
    else:
        sh_args = [
            os.path.join(NGRAM_BIN_PATH, "ngramprint"),
            "--ARPA",
            ngram_mod,
            ngram_arpa,
        ]
        return subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,)


def make_kenlm(ngram_arpa, force):
    ngram_kenlm = ngram_arpa + ".kenlm"
    if os.path.isfile(ngram_kenlm) and not force:
        print("File", ngram_kenlm, "exists. Skipping.")
        return
    else:
        sh_args = [KENLM_BIN_PATH, "trie", "-i", ngram_arpa, ngram_kenlm]
        return subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,)


def test_perplexity(mod_c, symbols, test_txt, nemo_model_file, tmp_path):
    test_far = os.path.join(tmp_path, os.path.split(test_txt)[1] + ".far")
    farcompile(symbols, test_txt, test_far, nemo_model_file, False, False, False)
    res_p = perplexity(mod_c, test_far)
    return res_p


def main(arpa_a, alpha, arpa_b, beta, out_path, test_file, symbols, nemo_model_file, force):

    mod_c, arpa_c = merge(arpa_a, alpha, arpa_b, beta, out_path, force)

    if test_file and nemo_model_file:
        if not symbols:
            symbols = os.path.join(out_path, os.path.split(nemo_model_file)[1] + ".syms")
            make_symbol_list(nemo_model_file, symbols, force)
        test_p = test_perplexity(mod_c, symbols, test_file, nemo_model_file, out_path)
        print("Test perplexity", test_p)

    print("Making ARPA model", arpa_c)
    print("\n", make_arpa(mod_c, arpa_c, force), "\n")
    print("\n", make_kenlm(arpa_c, force), "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="Avg pytorch weights")
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
        help="Path symbols file. Count perplexity if provided. Used as: symbols=earnest.syms",
    )
    parser.add_argument(
        "--nemo_model_file", required=False, type=str, default=None, help="Path to the arpa_b",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Whether to rewrite all files")
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(_parse_args()))
