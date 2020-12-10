
import os
import sys
from argparse import ArgumentParser
import subprocess
import numpy as np
from normalize import normalizers
from typing import List, Dict, Union, Tuple, Optional

'''
Runs normalization on text data
'''

def load_file(file_path: str) -> List[str]:
    """
    Load given text file into list of string.
    Args: 
        file_path: file path
    Returns: flat list of string
    """
    res = []
    with open(file_path, 'r') as fp:
        for line in fp:
            if line:
                res.append(line.strip())
    return res

def write_file(file_path: str, data: List[str]):
    """
    Writes out list of string to file.
    Args:
        file_path: file path
        data: list of string
    """
    with open(file_path, 'w') as fp:
        for line in data:
            fp.write(line+'\n')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", help="input file path", required=True, type=str)
    parser.add_argument("--output", help="output file path", required=True, type=str)
    parser.add_argument("--verbose", help="print normalization info. For debugging", action='store_true')
    parser.add_argument("--normalizer", default='nemo', help="normlizer to use (" + ", ".join(normalizers.keys()) + ")", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.input
    normalizer = normalizers[args.normalizer]

    print("Loading data: " + file_path)
    data = load_file(file_path)
    
    print("- Data: " + str(len(data)) + " sentences")
    normalizer_prediction = normalizer(data, verbose=args.verbose)
    print("- Normalized. Writing out...")
    write_file(args.output, normalizer_prediction)
    



    

    