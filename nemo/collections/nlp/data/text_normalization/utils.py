from tqdm import tqdm
from copy import deepcopy
from nltk import word_tokenize
from nemo.collections.nlp.data.text_normalization.constants import *

# Helper Functions
def read_data_file(fp):
    insts, w_words, s_words, classes = [], [], [], []
    # Read input file
    with open(fp, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            es = [e.strip() for e in line.strip().split('\t')]
            if es[0] == '<eos>':
                inst = (deepcopy(classes), deepcopy(w_words), deepcopy(s_words))
                insts.append(inst)
                # Reset
                w_words, s_words, classes = [], [], []
            else:
                classes.append(es[0])
                w_words.append(es[1])
                s_words.append(es[2])
    return insts

def normalize_str(input_str):
    input_str = ' '.join(word_tokenize(input_str.strip().lower()))
    input_str = input_str.replace('  ', ' ')
    return input_str
