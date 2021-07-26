from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import isdir, join

import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.nlp.data.text_normalization.utils import read_data_file

if __name__ == '__main__':
    parser = ArgumentParser(description='Combine multiple processed datasets (e.g., for multilingual training)')
    parser.add_argument('--input_dirs', action='append', help='Paths to folders of processed datasets', required=True)
    parser.add_argument('--output_dir', type=str, default='combined', help='Path to the output folder')
    args = parser.parse_args()

    # Create the output dir (if not exist)
    if not isdir(args.output_dir):
        mkdir(args.output_dir)

    # Read input datasets and combine them
    train, dev, test = [], [], []
    for split_name in constants.SPLIT_NAMES:
        if split_name == constants.TRAIN:
            cur_data = train
        if split_name == constants.DEV:
            cur_data = dev
        if split_name == constants.TEST:
            cur_data = test
        # Loop through each input directory
        for input_dir in args.input_dirs:
            input_fp = join(input_dir, f'{split_name}.tsv')
            insts = read_data_file(input_fp)
            cur_data.extend(insts)
    print('After combining the datasets:')
    print(f'len(train): {len(train)}')
    print(f'len(dev): {len(dev)}')
    print(f'len(test): {len(test)}')

    # Output
    for split_name in constants.SPLIT_NAMES:
        output_fp = join(args.output_dir, f'{split_name}.tsv')
        with open(output_fp, 'w+') as output_f:
            if split_name == constants.TRAIN:
                cur_data = train
            if split_name == constants.DEV:
                cur_data = dev
            if split_name == constants.TEST:
                cur_data = test
            for inst in cur_data:
                cur_classes, cur_tokens, cur_outputs = inst
                for c, t, o in zip(cur_classes, cur_tokens, cur_outputs):
                    output_f.write(f'{c}\t{t}\t{o}\n')
                output_f.write('<eos>\t<eos>\n')
