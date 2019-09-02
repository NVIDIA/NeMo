"""
Interface to Baidu's CTC decoders
from https://github.com/PaddlePaddle/DeepSpeech/decoders/swig
"""

import argparse
import multiprocessing
import os
import pickle
import sys

import numpy as np
import toml
from ctc_decoders import Scorer
from ctc_decoders import ctc_beam_search_decoder_batch
from nemo_asr.parts.dataset import Manifest


parser = argparse.ArgumentParser(
    description="CTC decoding and tuning with LM rescoring"
)
parser.add_argument("--mode", help="either 'eval' (default) or 'infer'",
                    default="eval")
parser.add_argument(
    "--model_toml", help="Toml file describing the model and vocabulary",
    required=True
)
parser.add_argument(
    "--infer_output_file", help="output CSV file for 'infer' mode",
    required=False
)
parser.add_argument("--logits", help="pickle file with CTC logits",
                    required=True)
parser.add_argument(
    "--labels",
    help="JSON file with audio filenames \
      (and ground truth transcriptions for 'eval' mode)",
    required=True,
)
parser.add_argument("--lm", help="KenLM binary file", required=True)
parser.add_argument("--alpha", type=float, help="value of LM weight",
                    required=True)
parser.add_argument(
    "--alpha_max",
    type=float,
    help="maximum value of LM weight (for a grid search in 'eval' mode)",
    required=False,
)
parser.add_argument(
    "--alpha_step",
    type=float,
    help="step for LM weight's tuning in 'eval' mode",
    required=False,
    default=0.1,
)
parser.add_argument(
    "--beta", type=float, help="value of word count weight", required=True
)
parser.add_argument(
    "--beta_max",
    type=float,
    help="maximum value of word count weight (for a grid search in \
      'eval' mode",
    required=False,
)
parser.add_argument(
    "--beta_step",
    type=float,
    help="step for word count weight's tuning in 'eval' mode",
    required=False,
    default=0.1,
)
parser.add_argument(
    "--beam_width",
    type=int,
    help="beam width for beam search decoder",
    required=False,
    default=128,
)
parser.add_argument(
    "--dump_all_beams_to",
    help="filename to dump all beams in eval mode for debug purposes",
    required=False,
    default="",
)
args = parser.parse_args()

if args.alpha_max is None:
    args.alpha_max = args.alpha
# include alpha_max in tuning range
args.alpha_max += args.alpha_step / 10.0

if args.beta_max is None:
    args.beta_max = args.beta
# include beta_max in tuning range
args.beta_max += args.beta_step / 10.0

num_cpus = multiprocessing.cpu_count()


def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b.
    The code was taken from: http://hetland.org/coding/python/levenshtein.py
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]


def load_dump(pickle_file):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    return data


def get_logits(data, labels):
    """
    Get logits from pickled data.
    There are two versions of pickle file (and data):
    1. raw logits NumPy array
    2. dictionary with logits and additional meta information
    """
    # Get a mapping to swap logits for blank
    if isinstance(data, np.ndarray) or isinstance(data, list):
        # convert NumPy array to dict format
        logits = {}
        for idx, line in enumerate(labels):
            audio_filename = line["audio_filepath"]
            logits[audio_filename] = data[idx]
    else:
        logits = data["logits"]
    return logits


def load_labels(csv_file, vocab):
    # labels = np.loadtxt(csv_file, skiprows=1, delimiter=',', dtype=str)
    return Manifest([csv_file], vocab, normalize=True).data


def load_vocab(model_toml):
    jasper_model_definition = toml.load(model_toml)
    vocab = jasper_model_definition["labels"]["labels"]
    return vocab


def greedy_decoder(logits, vocab, merge=True):
    s = ""
    c = ""
    for i in range(logits.shape[0]):
        c_i = vocab[np.argmax(logits[i])]
        if merge and c_i == c:
            continue
        s += c_i
        c = c_i
    if merge:
        s = s.replace("_", "")
    return s


def softmax(x):
    m = np.expand_dims(np.max(x, axis=-1), -1)
    e = np.exp(x - m)
    return e / np.expand_dims(e.sum(axis=-1), -1)


def evaluate_wer(logits, labels, vocab, decoder):
    total_dist = 0.0
    total_count = 0.0
    wer_per_sample = np.empty(shape=len(labels))

    empty_preds = 0
    for idx, line in enumerate(labels):
        audio_filename = line["audio_filepath"]
        label = "".join([vocab[k] for k in line["transcript"]])
        pred = decoder(logits[audio_filename], vocab)
        dist = levenshtein(label.lower().split(), pred.lower().split())
        if pred == "":
            empty_preds += 1
        total_dist += dist
        total_count += len(label.split())
        wer_per_sample[idx] = dist / len(label.split())
    print("# empty preds: {}".format(empty_preds))
    wer = total_dist / total_count
    return wer, wer_per_sample


data = load_dump(args.logits)
vocab = load_vocab(args.model_toml)
vocab.append("_")
labels = load_labels(args.labels, vocab)
logits = get_logits(data, labels)

probs_batch = []
for line in labels:
    audio_filename = line["audio_filepath"]
    probs_batch.append(softmax(logits[audio_filename]))

if args.mode == "eval":
    wer, _ = evaluate_wer(logits, labels, vocab, greedy_decoder)
    print("Greedy WER = {:.4f}".format(wer))
    best_result = {"wer": 1e6, "alpha": 0.0, "beta": 0.0, "beams": None}
    for alpha in np.arange(args.alpha, args.alpha_max, args.alpha_step):
        for beta in np.arange(args.beta, args.beta_max, args.beta_step):
            scorer = Scorer(alpha, beta, model_path=args.lm,
                            vocabulary=vocab[:-1])
            res = ctc_beam_search_decoder_batch(
                probs_batch,
                vocab[:-1],
                beam_size=args.beam_width,
                num_processes=num_cpus,
                ext_scoring_func=scorer,
            )
            total_dist = 0.0
            total_count = 0.0
            for idx, line in enumerate(labels):
                label = "".join([vocab[k] for k in line["transcript"]])
                score, text = [v for v in zip(*res[idx])]
                pred = text[0]
                dist = levenshtein(label.lower().split(), pred.lower().split())
                total_dist += dist
                total_count += len(label.split())
            wer = total_dist / total_count
            if wer < best_result["wer"]:
                best_result["wer"] = wer
                best_result["alpha"] = alpha
                best_result["beta"] = beta
                best_result["beams"] = res
            print("alpha={:.2f}, beta={:.2f}: WER={:.4f}".format(alpha, beta,
                                                                 wer))
    print(
        "BEST: alpha={:.2f}, beta={:.2f}, WER={:.4f}".format(
            best_result["alpha"], best_result["beta"], best_result["wer"]
        )
    )

    if args.dump_all_beams_to:
        with open(args.dump_all_beams_to, "w") as f:
            for beam in best_result["beams"]:
                f.write("B=>>>>>>>>\n")
                for pred in beam:
                    f.write("{} 0.0 0.0 {}\n".format(pred[0], pred[1]))
                f.write("E=>>>>>>>>\n")

elif args.mode == "infer":
    scorer = Scorer(args.alpha, args.beta, model_path=args.lm,
                    vocabulary=vocab[:-1])
    res = ctc_beam_search_decoder_batch(
        probs_batch,
        vocab[:-1],
        beam_size=args.beam_width,
        num_processes=num_cpus,
        ext_scoring_func=scorer,
    )
    infer_preds = np.empty(shape=(len(labels), 2), dtype=object)
    for idx, line in enumerate(labels):
        filename = line["audio_filepath"]
        score, text = [v for v in zip(*res[idx])]
        infer_preds[idx, 0] = filename
        infer_preds[idx, 1] = text[0]

    np.savetxt(
        args.infer_output_file,
        infer_preds,
        fmt="%s",
        delimiter=",",
        header="wav_filename,transcript",
    )
