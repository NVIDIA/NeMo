# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import argparse
import os

from nemo import logging


def __convert_data(in_file, out_text, out_labels, max_length):
    """
    in_file should be in the IOB format, see example here:
    https://www.clips.uantwerpen.be/conll2003/ner/.

    After the convertion, the dataset is splitted into 2 files: text.txt
    and labels.txt.
    Each line of the text.txt file contains text sequences, where words
    are separated with spaces. The labels.txt file contains corresponding
    labels for each word in text.txt, the labels are separated with spaces.
    Each line of the files should follow the format:
    [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and
    [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).

    """
    in_file = open(in_file, 'r')

    if max_length == -1:
        with open(out_text, 'w') as out_text, open(out_labels, 'w') as out_labels:
            for line in in_file:
                if line == '\n':
                    out_text.write(line)
                    out_labels.write(line)
                else:
                    line = line.split()
                    out_text.write(line[0] + ' ')
                    out_labels.write(line[-1] + ' ')

    else:
        lines = []
        words = []
        labels = []
        with open(out_text, 'w') as out_text, open(out_labels, 'w') as out_labels:
            lines = in_file.readlines()
            for line_id, line in enumerate(lines):
                logging.info(f"{line_id} {len(lines)}")
                contends = line.strip()
                if len(contends) == 0:
                    assert len(words) == len(labels)
                    if len(words) > max_length:
                        # split if the sentence is longer than 30
                        while len(words) > max_length:
                            tmplabel = labels[:max_length]
                            for iidx in range(len(tmplabel)):
                                if tmplabel.pop() == 'O':
                                    break
                            l = ' '.join([label for label in labels[: len(tmplabel) + 1] if len(label) > 0])
                            w = ' '.join([word for word in words[: len(tmplabel) + 1] if len(word) > 0])
                            # lines.append([l, w])
                            out_text.write(w + "\n")
                            out_labels.write(l + "\n")
                            words = words[len(tmplabel) + 1 :]
                            labels = labels[len(tmplabel) + 1 :]

                    if len(words) == 0:
                        continue
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    # lines.append([l, w])
                    out_text.write(w + "\n")
                    out_labels.write(l + "\n")
                    words = []
                    labels = []
                    continue

                word = line.strip().split()[0]
                label = line.strip().split()[-1]
                words.append(word)
                labels.append(label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert data from IOB '
        + 'format to the format compatible with '
        + 'nlp/examples/token_classification.py'
    )
    parser.add_argument("--data_file", required=True, type=str)
    parser.add_argument("--max_length", default=-1, type=int)
    args = parser.parse_args()

    data_dir = os.path.dirname(args.data_file)
    basename = os.path.basename(args.data_file)
    prefix, ext = os.path.splitext(basename)
    if not os.path.exists(args.data_file):
        raise FileNotFoundError("{data_file} not found in {data_dir}")

    logging.info(f'Processing {args.data_file}')
    out_text = os.path.join(data_dir, 'text_' + prefix + '.txt')
    out_labels = os.path.join(data_dir, 'labels_' + prefix + '.txt')

    __convert_data(args.data_file, out_text, out_labels, args.max_length)
    logging.info(f'Processing of the {args.data_file} is complete')
