import json
import os
from argparse import ArgumentParser
from glob import glob

from nemo_text_processing.g2p.data.data_utils import read_wikihomograph_file
from tqdm import tqdm


"""
Converts WikiHomograph data to .json manifest format for HeteronymClassificationModel training.
Details of the WikiHomograph data:
    https://github.com/google-research-datasets/WikipediaHomographData/tree/master/data/eval

"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_folder', help="Path to data folder with .tsv files", type=str, required=True)
    parser.add_argument("--output", help="Path to output .json file to store the data", type=str, required=True)
    return parser.parse_args()


def convert_wikihomograph_data_to_manifest(data_folder: str, output_manifest: str):
    """
    Convert WikiHomograph data to .json manifest

    Args:
        data_folder: data_folder that contains .tsv files
        output_manifest: path to output file
    """
    with open(output_manifest, "w") as f_out:
        for file in tqdm(glob(f"{data_folder}/*.tsv")):
            file_name = os.path.basename(file)
            sentences, start_end_indices, homographs, word_ids = read_wikihomograph_file(file)
            for i, sent in enumerate(sentences):
                start, end = start_end_indices[i]

                homograph = file_name.replace(".tsv", "")
                homograph_span = sent[start:end]
                if homograph_span.lower() != homograph and sent.lower().count(homograph) == 1:
                    start = sent.lower().index(homograph)
                    end = start + len(homograph)
                    homograph_span = sent[start:end].lower()
                    assert homograph == homograph_span.lower()

                assert homograph_span.lower() == homograph
                entry = {
                    "text_graphemes": sent,
                    "start_end": [start, end],
                    "homograph_span": homograph_span,
                    "word_id": word_ids[i],
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Data saved at {output_manifest}")


if __name__ == '__main__':
    args = parse_args()
    convert_wikihomograph_data_to_manifest(args.data_folder, args.output)
