# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


## Wikipedia titles taken from YAGO corpus. Preparation of this file is described in preprocess_yago.sh.
## Format: original title, and clean
##    Żywkowo,_Podlaskie_Voivodeship         zywkowo_podlaskie_voivodeship
##    Żywkowo,_Warmian-Masurian_Voivodeship  zywkowo_warmian-masurian_voivodeship
##    Żywocice                               zywocice
##    ZYX                                    zyx
##    Zyx_(cartoonist)                       zyx_cartoonist
##    ZyX_(company)                          zyx_company
YAGO_ENTITIES=yago.uniq2

## Preparation of this folder is described in preprocess_yago.sh. Its structure looks like this
##  ├── part_xaa.tar.gz
##  ├── ...
##  └── part_xeс.tar.gz
## Names do not matter, each tar.gz contains multiple downloaded articles, each in a separate json file 
WIKIPEDIA_FOLDER=../yago_wikipedia

## Articles with these titles will be skipped (as they are reserved for testing)
## To generate this file use ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/evaluation/get_all_titles_from_spoken_wikipedia.py --input_folder en/en/english --output_file spoken_wiki_titles.txt
EXCLUDE_TITLES=spoken_wiki_titles.txt

## Vocabulary of aligned YAGO subphrases, allows to use not only Wikipedia titles as whole phrases, but also their parts.
## Preparation of this file is described in get_ngram_mappings.sh.
SUBMISSPELLS=sub_misspells.txt

## Vocabulary of n-gram mappings
## Preparation of this file is described in get_ngram_mappings.sh.
NGRAM_MAPPINGS=replacement_vocab_filt.txt

## Vocabulary from Google Text Normalization Dataset.
## It is used here to perform a simple fast text normalization by substitution.
## To generate this file use ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/get_multi_reference_vocab.py
## Format: semiotic class, spoken, written, frequency
## CARDINAL        seventeen       17      103679
## CARDINAL        seventeen       xvii    2212
## DATE    nineteen eighties       1980s   57236
## DATE    nineteen eighties       1980's  546
## DATE    nineteen eighties       nineteen eighties       1
## CARDINAL        four hundred    400     28999
## CARDINAL        four hundred    400,    19
GTN_REFERENCE_VOCAB=gtn_reference_vocab.filt

## Generate a file with IDF (inverse document frequencies) for words and short phrases.
## It is used in later steps to filter out frequent phrases.
## Format: phrase, idf, number of documents in which phrase occured
## in the  0.32193097471545545     2305627
## was     0.32511695805297663     2298293
## of the  0.3559607516604808      2228487
## ...
## emmanuel episcopal church       13.586499828372018      4
## cornewall       13.586499828372018      4
## george bryan    13.586499828372018      4
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/get_idf_from_yago_wiki.py \
  --input_folder ${WIKIPEDIA_FOLDER} \
  --exclude_titles_file ${EXCLUDE_TITLES} \
  --yago_entities_file ${YAGO_ENTITIES} \
  --output_file idf.txt

## Find Yago entities and its subphrases in full text paragraphs of Wikipedia articles.
## Generates a large file yago_wiki.txt of format: list of words/phrases that occured in given paragraph, paragraph text (original case, not normalized).
## boardman        Boardman, Samuel Lane (1903). The Naturalist of the Saint Croix. Memoir of George A. Boardman. Bangor: Privately printed.
## gelclair;undiluted      Gelclair is usually used 3 times a day or as needed. It is usually diluted with water and rinsed around the mouth. It can be used undiluted where no water is available, and applied directly.
## gelclair;painkillers;mucositis       Gelclair does not numb the mouth and can be used in conjunction with other treatment options for managing oral mucositis, including antibacterial mouthwashes and painkillers.
## geographic coordinate conversion;geodetic datum      In geodesy, geographic coordinate conversion is defined as translation among different coordinate formats or map projections all referenced to the same geodetic datum. A geographic coordinate transformation is a translation among different geodetic datums. Both geographic coordinate conversion and transformation will be considered in this article.
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/prepare_sentences_from_yago_wiki.py \
  --input_folder ${WIKIPEDIA_FOLDER} \
  --exclude_titles_file ${EXCLUDE_TITLES} \
  --yago_entities_file ${YAGO_ENTITIES} \
  --sub_misspells_file ${SUBMISSPELLS} \
  --idf_file idf.txt \
  --output_file yago_wiki.txt

## Take a sample from 10 Gb yago_wiki.txt file.
## Sampling is controlled by parameters --each_n_line (skip other) and --max_count (skips paragraph if all its phrases already occured at least as many times)
## Phrase lists and paragraphs are written to separate files with equal number of lines 
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/sample_phrases.py \
  --input_name yago_wiki.txt \
  --max_count 10 \
  --each_n_line 30 \ 
  --output_phrases_name yago_wiki_sample.phrases \
  --output_paragraphs_name yago_wiki_sample.paragraphs

## Normalize paragraphs using substitution by GTN vocabulary (fast and simple).
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/normalize_by_gtn_vocab.py \
  --input_file yago_wiki_sample.paragraphs \
  --output_file yago_wiki_sample.paragraphs.norm \
  --tn_vocab ${GTN_REFERENCE_VOCAB}


## Build an index for phrases and subphrases to later find similar candidates. 
## This script will generate several independent portions of index.
## The next script get_related_phrases.py will process each portion independently, thus it won't find related phrases from other portions
## This is not ideal, but will provide randomness which is not bad.
## Because the goal is not to find all related phrases but some of them to serve as negative candidates
## ATTENTION: while indexing we use much stricter logprob threshold than at inference (because of too many phrases)
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/index_phrases.py \
  --input_file ${SUBMISSPELLS} \
  --output_file index.txt \
  --ngram_mapping ${NGRAM_MAPPINGS} \
  --min_log_prob -1.0 \
  --max_phrases_per_ngram 400 \
  --max_dst_freq 10000 \
  --input_portion_size 500000

## ATTENTION: edit depending on how many portions you get 
for part in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13"
do
    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/get_related_phrases.py --input_file index.txt.${part} --output_file related_phrases.${part}.txt
done

cat related_phrases.*.txt > related_phrases.txt

## The file related_phrases.txt will be used later to sample some negative candidates which are somewhat similar to correct ones
## Format: correct, similar, similarity score
## adelshoffen     adelshofen      10
## adelshoffen     adelshof        7
## adelshoffen     albertshofen    7
## adelshoffen     aiterhofen      6
## adelshoffen     waltenhofen     6
## adam_welsh      adam_welsh      9
## adam_welsh      adam_walsh      8
## adam_welsh      adam_welichowski        7
## adam_welsh      adam_wehsely-swiczinsky 7
## adam_welsh      adam_westall    6
## ab_welsh        ab_welsh        8
## ab_welsh        adam_welsh      6
## ab_welsh        alan_welsh      6

## Collect frequent word n-grams from plain text of paragraphs. They will be later compared to target phrases to sample false positive candidates.  
## Output is split into 5 files - each for different n-gram length.
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/get_ngrams_from_yago_wiki.py \
  --input_file yago_wiki_sample.paragraphs.norm \
  --output_file frequent_ngrams \
  --min_freq 50 \
  --max_ngram_len 5

## Get list of all target phrases with their frequencies (counts one occurrence per paragraph).
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/get_actual_phrases.py \
  --input_file yago_wiki_sample.phrases \
  --output_file actual_phrases.txt

for part in "1" "2" "3" "4" "5"
do
    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/index_phrases.py \
      --input_file frequent_ngrams.${part} \
      --output_file index.txt \
      --ngram_mapping ${NGRAM_MAPPINGS} \
      --min_log_prob -4.0 \
      --max_phrases_per_ngram 400 \
      --max_dst_freq 10000 \
      --input_portion_size 100000 \
      --input_max_lines 100000

    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/get_candidates.py \
      --index_file index.txt.0 \
      --input_file actual_phrases.txt \ 
      --output_file candidates.${part}.txt \ 
      --max_candidates 10 \
      --min_real_coverage 0.8 \ 
      --match_whole_input=true \
      --skip_empty=true \
      --skip_same=true
done

cat candidates.*.txt > reverse_frequent_ngrams_candidates.txt

## The file reverse_frequent_ngrams_candidates.txt is later used to sample false positives.
## Format: Yago phrase or subphrase, list of frequent n-grams somewhat similar to it
## issn    noises;cessna;meissen;issuing;hussein;giessen;poisson
## brussels        bruxelles;brasileira;brasileiro;russell's;roosevelt's;rescheduled;seuil;tallahassee;rousseau
## julia   giulio;juliet;giulia;yulia;julie;julien;julian;julio
## nevada  evade;evaded;weighed;daewoo;vidal;vedas;geneve;nineveh
## ferdinand       ferdinando;sandford;ordinance;sardinian;ordinances;ordained;endangered;verdun;wandered
## leeds   leads;scheduled;pleads;ruled;deeds;zayed;hauled
## gearld wright   during world war ii he

## Search phrases in the paragraph, cut spans containing some phrase(s) and some surrounding context.
## Two outputs: 1) examples with at least 1 correct candidate, 2) examples with no correct candidates.
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/get_fragments_from_yago_wiki.py \
  --input_phrases_file yago_wiki_sample.phrases \
  --input_paragraphs_file yago_wiki_sample.paragraphs.norm \
  --output_file_non_empty fragments_non_empty.txt \
  --output_file_empty fragments_empty.txt

## File fragments_non_empty.txt:
##   programming block known as cdis curriculum development institute of singapore began     1 2     [cdis] 27 31;[curriculum development] 32 54
##   tibensky january nineteen eighty four matej bel zivot a 1 2 3 4 [tibensky] 0 8;[matej] 38 43;[bel] 44 47;[zivot] 48 53
##   in response to malaysia's tv pendidikan 1 2 3   [malaysia's] 15 25;[tv pendidikan] 26 39;[pendidikan] 29 39

## File fragments_empty.txt:
##   on the first of february of that year   0
##   first of february of that year  0
##   celebrated its thirty years of television       0
##   its thirty years of television broadcasting on  0

## Add misspells and 10 candidates via different sampling
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/construct_positive_and_negative_examples.py \
  --non_empty_fragments_file fragments_non_empty.txt \
  --empty_fragments_file fragments_empty.txt \
  --related_phrases_file related_phrases.txt \
  --sub_misspells_file ${SUBMISSPELLS} \
  --reverse_frequent_ngrams_candidates reverse_frequent_ngrams_candidates.txt \
  --output_file_positive positive.txt \
  --output_file_negative negative.txt

## File positive.txt:
## Format: text fragment, candidates (# - similar to correct candidate, * - false positive, & - random, no mark - correct), indices of correct targets in candidates, positions of correct targets in text,  misspelled targets
##   twenty thirteen top marques monaco      #marquet;monaco;marques;*mark oaten;*tpr;#aldo bonacossa;*mark kligman;*patwin;#alexandre buonacorsi;*theo martins  2 3      28 34;20 27     monoco;marx
##   the collection of the museum of tuscan  #escanlar;tuscan;*over the hump;*todd santos;&samtiden;*glenbow museum;*mimara museum;#puscanski;#hugh scanlon;*museum attendant     2       32 38   tusk can

## File negative.txt:
## Format: text fragment, candidates (* - false positive, & - random)
##   tom abbott played from nineteen seventy five seventy eight      *framke;*planeten;*nineta gusti;*pikfyve;*platycerioideae;*five guys;*live seventy nine;*japanese night heron;*ivantsovi;*abbott drive
##   assistant offensive coach for penn      *vasile stanescu;*jessica stern;*tet offensive;*pend;*dunsbach ferry;*hambach forest;*poochhe;*may offensive;*sven oftedal;*jessie stanton

## Make training examples as expected by the neural model.
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/make_final_training_examples.py \
  --positive_file positive.txt \
  --negative_file negative.txt \
  --output_file all.tsv

## bug fix
#grep -v "ʹ" all.tsv > all2.tsv
#mv all2.tsv > all.tsv

## File all.tsv contains all training examples - one example per line.
## Format: asr-hypothesis split by characters, 10 candidates, indices of correct candidates, positions of misspeled fragments in asr-hypothesis
## t _ v _ o r _ e y e _ o n _ t h e _ f i r s t _ o f _ f e b r u a r y _ o f _ t h a t _ y e a r _ s _ b _ c _ c e l e b r a t e d       t v r _ t u s c a n;t h e _ f i r s t _ e d e n;w s p c;s b c _ p u b l i s h i n g _ b u i l d i n g;t v r _ g r a n t u r a;t v r i;s b c;s o a e b _ t a i;d u y a r;f i n t h e n     6 7     CUSTOM 0 10;CUSTOM 49 54
## d i s t r i c t _ o f _ a f g h a n i s t a n   d i s t r i c t _ o f _ p r i s t i n a;f e n i _ d i s t r i c t;d i s t r i c t _ o f _ l o u i s i a n a;m o r g a n _ s t a n l e y;f l a g _ o f _ a f g h a n i s t a n;u n a _ d i s t r i c t;z i r c _ d i s t r i c t;f i n n _ f u g l e s t a d;p h l o _ f i n i s t e r;g u s t a f _ a u l e n     0

## To generate files config.json, label_map.txt, semiotic_classes.txt, run generate_configs.sh

## HOW TO TRAIN WITH NON-TARRED DATA 
## You can take any subsets of all.tsv to use directly as training and validation datasets.
## Example of all files needed to run training with non-tarred data:
## data_folder_example
##   ├── config.json
##   ├── label_map.txt
##   ├── semiotic_classes.txt
##   ├── test.tsv
##   └── train.tsv

## To run training with non-tarred data, use ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/run_training.sh
## Note that training with non-tarred data only works on single gpu. It makes sense if you use 1-2 million examples or less.

## HOW TO TRAIN WITH TARRED DATA
## To convert data to tarred format, split all.tsv to pieces of 110'000 examples and use ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/convert_data_to_tarred.sh
## To run training with tarred data, use ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/run_training_tarred.sh
## data_folder_example
##   ├── train_tarred
##   |   ├── part1.tar
##   |   ├── ...
##   |   └── part200.tar
##   ├── config.json
##   ├── label_map.txt
##   ├── semiotic_classes.txt
##   └── test.tsv
