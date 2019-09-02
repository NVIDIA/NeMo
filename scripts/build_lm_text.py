import pandas as pd
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build N-gram LM model from text file')
    parser.add_argument('text', metavar='text', type=str,
                        help='text file')
    parser.add_argument('--n', type=int, help='n for n-grams', default=3)
    args = parser.parse_args()

    path_prefix, _ = os.path.splitext(args.text)
    corpus_name = args.text
    arpa_name = path_prefix + '.arpa'
    lm_name = path_prefix + '-lm.binary'

    lmplz_tmp = 'decoders/kenlm/build/bin/lmplz --text {} --arpa {} --o {}'
    command = lmplz_tmp.format(corpus_name, arpa_name, args.n)
    print(command)
    os.system(command)

    tmp = 'decoders/kenlm/build/bin/build_binary trie -q 8 -b 7 -a 256 {} {}'
    command = tmp.format(arpa_name, lm_name)
    print(command)
    os.system(command)
