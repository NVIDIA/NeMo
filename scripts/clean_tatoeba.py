import os
import random

def replace(data_dir, in_file, out_file):
    verbs = ['must', 'may', 'might']
    add_ons = ['He', 'She', 'Maggy', 'Sam', 'Peter']
    count = 0

    in_file = open(os.path.join(data_dir, in_file), 'r')
    out_file = open(os.path.join(data_dir, out_file), 'w')

    for line in in_file:
        line = line.strip().split()
        if len(line) > 3:
            if (line[0] == 'I' and line[1][-2:] == 'ed') or (line[0] == 'You' and line[1] in verbs):
                sub = add_ons[random.randint(0, len(add_ons) - 1)] + 'I'
                line[0] = sub
                out_file.write(' '.join(line) + '\n')
                count += 1
    print ('replaced:', count)


if __name__ == '__main__':
    data_dir = '/home/ebakhturina/data/tutorial_punct/dataset/new_format'
    replace(data_dir, 'clean_all_eng_sentences.txt', 'I_clean_all_eng_sentences.txt')
