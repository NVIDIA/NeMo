import glob
import os
import sys
from copy import deepcopy
from argparse import ArgumentParser
import multiprocessing
import glob

import inflect
import regex as re
from tqdm import tqdm

import wordninja

parser = ArgumentParser(description="TN cleaning")
parser.add_argument("--output_dir", required=True, type=str, help='Path to output directory')
parser.add_argument("--input_dir", required=True, type=str, help='Path to input file')
parser.add_argument("--size", default=multiprocessing.cpu_count(), type=int, help='cpu count')
args = parser.parse_args()
engine = inflect.engine()

def process_url(o):
    """
    The function is used to process the spoken form of every URL in an example.
    E.g., "dot h_letter  _letter t_letter  _letter m_letter  _letter l_letter" ->
          "dot h t m l"

    Args:
        tokens: The tokens of the written form
        outputs: The expected outputs for the spoken form
        lang: Selected language.
    Return:
        outputs: The outputs for the spoken form with preprocessed URLs.
    """
    
    def flatten(l):
        """ flatten a list of lists """
        return [item for sublist in l for item in sublist]

    if o != '<self>' and '_letter' in o:
        o_tokens = o.split(' ')
        all_spans, cur_span = [], []
        for j in range(len(o_tokens)):
            if len(o_tokens[j]) == 0:
                continue
            if o_tokens[j] == '_letter':
                all_spans.append(cur_span)
                all_spans.append([' '])
                cur_span = []
            else:
                o_tokens[j] = o_tokens[j].replace('_letter', '')
                cur_span.append(o_tokens[j])
        if len(cur_span) > 0:
            all_spans.append(cur_span)
        o_tokens = flatten(all_spans)

        o = ''
        for o_token in o_tokens:
            if len(o_token) > 1:
                o += ' ' + o_token + ' '
            else:
                o += o_token
        o = o.strip()
        o_tokens = wordninja.split(o)
        o = ' '.join(o_tokens)

    return o


digit = "0123456789"

num_words = list(range(0, 20)) + list(range(20, 100, 10))
num_words = (
    [engine.number_to_words(x, zero="zero").replace("-", " ").replace(",", "") for x in num_words]
    + ["hundred", "thousand", "million", "billion", "trillion"]
    + ["point"]
    + ["zero"]
)
in_mani = sys.argv[1]
out_mani = sys.argv[2]

LENGTH = 5

EN_GREEK_TO_SPOKEN = {
    'Τ': 'tau',
    'Ο': 'omicron',
    'Δ': 'delta',
    'Η': 'eta',
    'Κ': 'kappa',
    'Ι': 'iota',
    'Θ': 'theta',
    'Α': 'alpha',
    'Σ': 'sigma',
    'Υ': 'upsilon',
    'Μ': 'mu',
    'Χ': 'chi',
    'Π': 'pi',
    'Ν': 'nu',
    'Λ': 'lambda',
    'Γ': 'gamma',
    'Β': 'beta',
    'Ρ': 'rho',
    'τ': 'tau',
    'υ': 'upsilon',
    'φ': 'phi',
    'α': 'alpha',
    'λ': 'lambda',
    'ι': 'iota',
    'ς': 'sigma',
    'ο': 'omicron',
    'σ': 'sigma',
    'η': 'eta',
    'π': 'pi',
    'ν': 'nu',
    'γ': 'gamma',
    'κ': 'kappa',
    'ε': 'epsilon',
    'β': 'beta',
    'ρ': 'rho',
    'ω': 'omega',
    'χ': 'chi',
}
def int2num(digits):
    res = []
    for i, x in enumerate(digits):
        if x in digit:
            res.append(engine.number_to_words(str(x),  zero="zero").replace("-", " ").replace(",", ""))
        elif x == ".":
            res.append("point")
        elif x in [" ", ","]:
            continue
        else:
            print(f"remove {digits[:i]} from {digits[i:]}")
            break
    return " ".join(res)


def convert(example):
    cls, written, spoken = example

    written = re.sub(" ½", " 1/2", written)
    written = re.sub(" ⅓", " 1/3", written)
    written = re.sub(" ⅔", " 2/3", written)
    written = re.sub(" ¼", " 1/4", written)
    written = re.sub(" ¾", " 3/4", written)
    written = re.sub(" ⅕", " 1/5", written)
    written = re.sub(" ⅖", " 2/5", written)
    written = re.sub(" ⅗", " 3/5", written)
    written = re.sub(" ⅘", " 4/5", written)
    written = re.sub(" ⅙", " 1/6", written)
    written = re.sub(" ⅚", " 5/6", written)
    written = re.sub(" ⅛", " 1/8", written)
    written = re.sub(" ⅜", " 3/8", written)
    written = re.sub(" ⅝", " 5/8", written)
    written = re.sub(" ⅞", " 7/8", written)

    
    written = re.sub("^½", "1/2", written)
    written = re.sub("^⅓", "1/3", written)
    written = re.sub("^⅔", "2/3", written)
    written = re.sub("^¼", "1/4", written)
    written = re.sub("^¾", "3/4", written)
    written = re.sub("^⅕", "1/5", written)
    written = re.sub("^⅖", "2/5", written)
    written = re.sub("^⅗", "3/5", written)
    written = re.sub("^⅘", "4/5", written)
    written = re.sub("^⅙", "1/6", written)
    written = re.sub("^⅚", "5/6", written)
    written = re.sub("^⅛", "1/8", written)
    written = re.sub("^⅜", "3/8", written)
    written = re.sub("^⅝", "5/8", written)
    written = re.sub("^⅞", "7/8", written)

    written = re.sub("-½", "-1/2", written)
    written = re.sub("-⅓", "-1/3", written)
    written = re.sub("-⅔", "-2/3", written)
    written = re.sub("-¼", "-1/4", written)
    written = re.sub("-¾", "-3/4", written)
    written = re.sub("-⅕", "-1/5", written)
    written = re.sub("-⅖", "-2/5", written)
    written = re.sub("-⅗", "-3/5", written)
    written = re.sub("-⅘", "-4/5", written)
    written = re.sub("-⅙", "-1/6", written)
    written = re.sub("-⅚", "-5/6", written)
    written = re.sub("-⅛", "-1/8", written)
    written = re.sub("-⅜", "-3/8", written)
    written = re.sub("-⅝", "-5/8", written)
    written = re.sub("-⅞", "-7/8", written)

    written = re.sub("([0-9])\s?½", "\\1 1/2", written)
    written = re.sub("([0-9])\s?⅓", "\\1 1/3", written)
    written = re.sub("([0-9])\s?⅔", "\\1 2/3", written)
    written = re.sub("([0-9])\s?¼", "\\1 1/4", written)
    written = re.sub("([0-9])\s?¾", "\\1 3/4", written)
    written = re.sub("([0-9])\s?⅕", "\\1 1/5", written)
    written = re.sub("([0-9])\s?⅖", "\\1 2/5", written)
    written = re.sub("([0-9])\s?⅗", "\\1 3/5", written)
    written = re.sub("([0-9])\s?⅘", "\\1 4/5", written)
    written = re.sub("([0-9])\s?⅙", "\\1 1/6", written)
    written = re.sub("([0-9])\s?⅚", "\\1 5/6", written)
    written = re.sub("([0-9])\s?⅛", "\\1 1/8", written)
    written = re.sub("([0-9])\s?⅜", "\\1 3/8", written)
    written = re.sub("([0-9])\s?⅝", "\\1 5/8", written)
    written = re.sub("([0-9])\s?⅞", "\\1 7/8", written)

    
    written = re.sub("é", "e", written)

    
    written = re.sub("²", "2", written)
    written = re.sub("³", "3", written)

    if cls in ["TIME"]:
        written = re.sub("([0-9]): ([0-9])", "\\1:\\2", written)
    if cls == "MEASURE":
        written = re.sub("([0-9])\s?''", '\\1"',written)
    example[1] = written
    spoken = process_url(spoken)


    
    if cls in ["TELEPHONE", "DIGIT", "MEASURE", "DECIMAL", "MONEY", "ADDRESS"]:
        spoken = re.sub(" o ", " zero ", spoken)
        spoken = re.sub(" o ", " zero ", spoken)
        spoken = re.sub(" o ", " zero ", spoken)
        spoken = re.sub(" o ", " zero ", spoken)
        spoken = re.sub(" o ", " zero ", spoken)
        spoken = re.sub("^o ", "zero ", spoken)
        spoken = re.sub(" o$", " zero", spoken)
        spoken = re.sub("^sil ", "", spoken)
        spoken = re.sub(" sil ", " ", spoken)
        spoken = re.sub(" sil ", " ", spoken)
        spoken = re.sub(" sil ", " ", spoken)
        spoken = re.sub(" sil ", " ", spoken)
        spoken = re.sub(" sil ", " ", spoken)
        spoken = re.sub(" sil$", "", spoken)

    example[2] = spoken


    l = LENGTH - 3
    if not re.search("[0-9]{%s}[,\s]?[0-9]{3}" % l, written):
        if cls != "FRACTION":
            return
        
        idx = written.index("/")
        denominator = written[idx + 1 :].strip()
        if not re.search(r"[0-9]{4}", denominator):
            return
        else:
            print(written)

        

    if re.search("[0-9]{%s}\s[0-9]{3}" % l, written):
        print(f"more found {cls}\t{written}\t{spoken}")
    if cls == "CARDINAL":

        if written[0] == "-":
            digits = "minus " + int2num(written[1:])
        else:
            digits = int2num(written)

        spoken = digits

    elif cls == "ADDRESS":

        res = []
        if written[0] == "-":
            res.append("minus")
            written = written[1:]
        idx = re.search("[0-9]", written).start()
        number = int2num(written[idx:].strip())
        s_words = spoken.split()
        for i, x in enumerate(s_words):
            if x in num_words:
                break
        spoken = " ".join(s_words[:i]) + " " + number
        if res:
            spoken = "minus " + spoken

    elif cls == "DECIMAL":
        res = []
        for i, x in enumerate(written):
            if i == 0 and x == "-":
                res.append("minus")
            elif x in digit:
                res.append(engine.number_to_words(str(x), zero="zero").replace("-", " ").replace(",", ""))
            elif x == ".":
                res.append("point")
        spoken = " ".join(res)
        m = re.search("([a-z]+)", written)
        if m:
            spoken += " " + m.group(1)

    elif cls == "FRACTION":
        res = []
        if written[0] == "-":
            res.append("minus")
            written = written[1:]
        idx = written.index("/")
        numerator = written[:idx].strip()
        denominator = written[idx + 1 :].strip()
        if len(numerator) >= LENGTH:
            numerator = int2num(numerator)
        else:
            numerator = engine.number_to_words(str(numerator), zero="zero").replace("-", " ").replace(",", "")
        if len(denominator) >= 4:
            denominator = int2num(denominator)
        else:
            denominator = engine.number_to_words(str(denominator),zero="zero").replace("-", " ").replace(",", "")

        spoken = numerator + " slash " + denominator
        if res:
            spoken = "minus " + spoken

    elif cls == "MEASURE":

        res = []
        if written[0] == "-":
            res.append("minus")
            written = written[1:]
        try:
            idx = re.search("(?s:.*)([0-9]\s?[a-zA-Zµμ\/%Ω'])", written).end()
        except:
            import ipdb

            ipdb.set_trace()
        number = int2num(written[:idx].strip())
        s_words = spoken.split()
        for i, x in enumerate(s_words):
            if x not in num_words:
                break
        spoken = number + " " + " ".join(s_words[i:])
        if res:
            spoken = "minus " + spoken

    elif cls == "MONEY":
        res = []
        if written[0] == "-":
            res.append("minus")
            written = written[1:]
        idx = re.search("[0-9]", written).start()
        m = re.search("\.", written[idx:])
        idx_end = len(written)
        if m:
            idx_end = m.start() + idx
        number = int2num(written[idx:idx_end].strip())
        s_words = spoken.split()
        for i, x in enumerate(s_words):
            if x not in num_words:
                break
        spoken = number + " " + " ".join(s_words[i:])
        if res:
            spoken = "minus " + spoken

    elif cls == "ORDINAL":
        res = []
        if written[0] == "-":
            res.append("minus")
            written = written[1:]

        if "th" in written.lower():
            idx = written.lower().index("th")
        elif "rd" in written.lower():
            idx = written.lower().index("rd")
        elif "nd" in written.lower():
            idx = written.lower().index("nd")
        elif "st" in written.lower():
            idx = written.lower().index("st")
        
        if re.search(r"[¿¡ºª]", written) is None:
            try:
                spoken = int2num(written[:idx].strip()) + " " + written[idx:].lower()
            except:
                import ipdb

                ipdb.set_trace()
        if res:
            spoken = "minus " + spoken


    example[2] = spoken


def ignore(example):
    cls, written, spoken = example
    if cls in ["PLAIN", "LETTERS", "ELECTRONIC", "VERBATIM", "PUNCT"]:
        example[2] = "<self>"
    
    return example


def read_data_file(fp):
    """ Reading the raw data from a file of NeMo format
    For more info about the data format, refer to the
    `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.

    Args:
        fp: file paths
    Returns:
        insts: List of sentences parsed as list of words
    """

    file_name = fp.split("/")[-1]
    
    output_path = f"{args.output_dir}/{file_name}"
    print("-----output_path--------", output_path)

    insts, w_words, s_words, classes = [], [], [], []
    # Read input file

    delete_sentence = False
    with open(fp, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            es = [e.strip() for e in line.strip().split('\t')]
            if es[0] == '<eos>':
                if not delete_sentence:
                    inst = (classes, w_words, s_words)
                    insts.append(inst)
                # Reset
                w_words, s_words, classes = [], [], []
                
                delete_sentence = False
            else:
                convert(es)

                ignore(es)

                characters_ignore = "¿¡ºª"+"".join(EN_GREEK_TO_SPOKEN.keys())
                if re.search(rf"[{characters_ignore}]", es[1]) is not None:
                    delete_sentence = True
                    # print(f"delete {es[1]}")
                if re.search(r'[\u4e00-\u9fff]+', es[1]) is not None:
                    delete_sentence = True
                    # print(f"CJK {es[1]}")



                classes.append(es[0])
                w_words.append(es[1])
                s_words.append(es[2])

        inst = (classes, w_words, s_words)
        insts.append(inst)

    output_f = open(output_path, 'w+', encoding='utf-8')
    for ix, inst in enumerate(insts):
        cur_classes, cur_tokens, cur_outputs = inst
        for c, t, o in zip(cur_classes, cur_tokens, cur_outputs):
            output_f.write(f'{c}\t{t}\t{o}\n')

        output_f.write(f'<eos>\t<eos>\n')




def main():
    # input_files = glob.glob(f"{args.input_dir}/output-*")

    # with multiprocessing.Pool(args.size) as pool:
    #     pool.map(read_data_file, input_files)
    read_data_file(args.input_dir)

if __name__=="__main__":
    main()