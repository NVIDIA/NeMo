# Copyright (c) 2019 NVIDIA Corporation
#
# USAGE: python process_asr_text_tokenizer.py --manifest=<path to train manifest files, seperated by commas> \
#         --data_root="<output directory>" \
#         --vocab_size=<number of tokens in vocabulary> \
#         --tokenizer=<"bpe" or "wpe"> \
#         --log
# where <manifest> can be: train_clean_100, train_clean_360, train_other_500
# You can also put more than one data_set comma-separated:
# --manifest="train_clean_100,train_clean_360,train_other_500"
import argparse
import json
import logging
import os

import sentencepiece
import tokenizers

parser = argparse.ArgumentParser(description='LibriSpeech Data download')
parser.add_argument("--manifest", required=True, default=None, type=str, help='Comma separated list of manifest files')
parser.add_argument("--data_root", required=True, default=None, type=str, help='Output directory')
parser.add_argument("--vocab_size", default=1024, type=int, help='Vocabulary size')
parser.add_argument("--tokenizer", default="bpe", choices=["bpe", "wpe"], help='Type of tokenization to perform')
parser.add_argument("--log", action='store_true')
parser.set_defaults(log=False)
args = parser.parse_args()


def __build_document_from_manifests(
    data_root: str, manifests: str,
):
    if ',' in manifests:
        manifests = manifests.split(',')
    else:
        manifests = [manifests]

    document_dir = os.path.join(data_root, 'librispeech_doc')
    if not os.path.exists(document_dir):
        os.makedirs(document_dir)

    document_path = os.path.join(document_dir, 'document.txt')

    if os.path.exists(document_path):
        logging.info('Corpus already exists at path : %s', document_path)
        return document_path

    num_lines = 0
    with open(document_path, 'w') as out_writer:
        for manifest in manifests:
            with open(manifest, 'r') as in_reader:
                for line in in_reader:
                    item = json.loads(line)
                    text = item['text']

                    out_writer.write(text + '\n')
                    out_writer.flush()

                    num_lines += 1

            logging.info(f"Finished extracting manifest : {manifest}")

        logging.info("Finished extracting all manifests ! Number of sentences : {}".format(num_lines))
    return document_path


def __process_data(text_path: str, dst_folder: str, vocab_size: int, tokenizer_type: str):
    """
    Converts flac to wav and build manifests's json
    Args:
        text_path: source with text lines
        dst_folder: where wav files will be stored
        vocab_size: vocabular size used in encoding the text
        tokenizer_type: type of tokenization to perform - bpe or wpe
    Returns:
    """
    tokenizer_dir = os.path.join(dst_folder, 'librispeech_tokenizer_{}_v{}').format(tokenizer_type, vocab_size)

    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)

    if tokenizer_type == 'bpe':
        if os.path.exists(os.path.join(tokenizer_dir, 'tokenizer.model')):
            logging.warning("Model file already exists, overriding old model file !")
            os.remove(os.path.join(tokenizer_dir, 'tokenizer.model'))

        cmd = (
            f"--input={text_path} --model_prefix={tokenizer_dir}/tokenizer "
            f"--vocab_size={vocab_size} "
            f"--shuffle_input_sentence=true --hard_vocab_limit=false "
            f"--bos_id=-1 --eos_id=-1 "
            f"--normalization_rule_name=nmt_nfkc_cf"
        )

        sentencepiece.SentencePieceTrainer.Train(cmd)

        # Add BERT control symbols
        tokens = []

        with open(f"{tokenizer_dir}/tokenizer.vocab", "r") as f:
            f.readline()  # skip first <unk> token

            # Read tokens from each line and parse for vocab
            for line in f:
                piece = line.split("\t")[0]
                token = piece[1:] if piece.startswith("▁") else f"##{piece}"

                if len(token) > 0:
                    tokens.append(token)
                else:
                    # Assume token is just "_" and insert
                    tokens.append("_")

        vocab = tokens

        # Save vocabulary to output file
        vocab_file = f'{tokenizer_dir}/vocab.txt'
        with open(vocab_file, "w") as f:
            for token in vocab:
                f.write(f"{token}\n".format())

    else:
        tokenizer = tokenizers.BertWordPieceTokenizer(lowercase=True)

        tokenizer.train(text_path, vocab_size=vocab_size)
        tokenizer.save(tokenizer_dir)

    return tokenizer_dir


def main():
    data_root = args.data_root
    manifests = args.manifest
    vocab_size = args.vocab_size
    tokenizer = args.tokenizer

    data_root = os.path.join(data_root, 'LibriSpeechTokenizer')

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if args.log:
        logging.basicConfig(level=logging.INFO)

    text_corpus_path = __build_document_from_manifests(data_root, manifests)
    tokenizer_path = __process_data(text_corpus_path, data_root, vocab_size, tokenizer)

    print("Serialized tokenizer at location :", tokenizer_path)
    logging.info('Done!')


if __name__ == "__main__":
    main()
