#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys

def get_questions_from_squad(data_dir, output_dir):
    """
    Extracts questions from the SQuAD dataset
    """
    files = ['dev-v2.0.json', 'train-v2.0.json']

    out_file = os.path.join(output_dir, 'questions_squad.txt')
    out_file = open(out_file, 'w')

    for file in files:
        in_file = os.path.join(data_dir, file)
        if not os.path.exists(in_file):
            raise FileNotFoundError(f'{in_file} not found.')

        data = json.load(open(in_file, 'r'))['data']
                
        for topic in data:
            for example in topic['paragraphs']:
                for qas in example['qas']:
                    out_file.write(qas['question'] + '\n')
 
    print (f'Questions from SQuAD extracted to {out_file.name}.')


def get_questions_from_coqa(data_dir, output_dir):
    """
    Extracts questions from the SQuAD dataset
    """
    files = ['coqa-train-v1.0.json', 'coqa-dev-v1.0.json']

    out_file = os.path.join(data_dir, 'questions_coqa.txt')
    out_file = open(out_file, 'w')

    for file in files:
        in_file = os.path.join(data_dir, file)
        if not os.path.exists(in_file):
            raise FileNotFoundError(f'{in_file} not found.')

        data = json.load(open(in_file, 'r'))['data']
        
        for topic in data:
            for questions in topic['questions']:
                question = questions['input_text'].capitalize()
                
                # check that the quesiont ends with ?
                # and there is no space between ? and the last word
                if question[-1] == '?':
                    question = question[:-1].strip() + '?'
                else:
                    question = question.strip() + '?'
                
                out_file.write(question + '\n')


def get_questions_and_periods_from_tatoeba(data_dir, output_dir):
    
    in_file = os.path.join(data_dir, 'english_sentences.csv')

    if not os.path.exists(in_file):
        raise FileNotFoundError(f'{in_file} not found.')

    in_file = open(in_file, 'r')
    out_file_questions = open(os.path.join(output_dir, 'questions_tatoeba.txt'), 'w')
    out_file_periods = open(os.path.join(output_dir, 'periods_no_commas_tatoeba.txt'), 'w')

    for line in in_file:
        line = line.split('\t')[2].strip()

        # extract question mark from tatoeba data
        if len(line) > 0 and line[-1] == '?' and re.match('^[A-Z][a-z.,?\s]+$', line) is not None:
            out_file_questions.write(line + '\n')

        #extract sentences with only periods - no commas, etc
        elif len(line) > 0 and line[-1] == '.' and re.match('^[A-Z][a-z.\s]+$', line) is not None:
            out_file_periods.write(line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a punctuation dataset")
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str,
        choices=['squad', 'coqa', 'tatoeba'])
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f'{args.data_dir} not found.')

    print(f"Processing {args.dataset}")
    function_name = 'get_questions_and_periods_from_' if args.dataset == 'tatoeba' else 'get_questions_from_'
    result = getattr(sys.modules[__name__], function_name + args.dataset)(args.data_dir, args.output_dir)
















































