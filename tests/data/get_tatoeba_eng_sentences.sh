#!/bin/bash
mkdir tatoeba_data
cd tatoeba_data
wget https://downloads.tatoeba.org/exports/sentences.csv
grep -P "\teng\t" sentences.csv > english_sentences.csv
head -n 900000 english_sentences.csv > train_sentences.csv
tail -n +900001 english_sentences.csv > eval_sentences.csv
rm english_sentences.csv
rm sentences.csv
