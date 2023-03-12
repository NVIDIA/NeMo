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


NEMO_PATH=/home/aleksandraa/nemo

## download yagoTypes.tsv from https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/

awk 'BEGIN {FS="\t"} {print $2}' < yagoTypes.tsv | sort -u > yago.uniq
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/preprocess_yago.py --input_name yago.uniq --output_name yago.uniq2
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/extract_single_words_from_yago.py --input_name yago.uniq2 --output_name yago.vocab.txt

## Now we have two files
## 1) yago.uniq2 with format: original title, and clean
##    Żywkowo,_Podlaskie_Voivodeship         zywkowo_podlaskie_voivodeship
##    Żywkowo,_Warmian-Masurian_Voivodeship  zywkowo_warmian-masurian_voivodeship
##    Żywocice                               zywocice
##    ZYX                                    zyx
##    Zyx_(cartoonist)                       zyx_cartoonist
##    ZyX_(company)                          zyx_company
##
## 2) yago.vocab.txt that can be used later for g2p inference
##    c a r t o o n i s t
##    c o m p a n y
##    m a s u r i a n
##    p o d l a s k i e
##    v o i v o d e s h i p
##    w a r m i a n
##    z y w k o w o
##    z y w o c i c e
##    z y x

## We want to download all Wikipedia articles with titles from yago.uniq2
## Example of download command
## wget "https://en.wikipedia.org/w/api.php?format=xml&action=query&prop=extracts&titles=Anna_Dumitriu&redirects=true&format=json&explaintext=1&exsectionformat=plain" -O anna_dumitriu.txt

## Use the below command or split into several parts:
## awk 'BEGIN {FS="\t"; print "#!/usr/bin/env bash"} {print "wget \"https://en.wikipedia.org/w/api.php?format=xml&action=query&prop=extracts&titles=" $1 "&redirects=true&format=json&explaintext=1&exsectionformat=plain\" -O \"articles/" $2 ".txt\"\nsleep 0.1"}' < yago.uniq2 > run_wget.sh
## ./run_wget.sh

## To use downloaded artiles in later scripts, you need to create a folder with following structure:
## yagowiki
##  ├── part_xaa.tar.gz
##  ├── ...
##  └── part_xeс.tar.gz
## Names do not matter, each tar.gz contains multiple downloaded articles, each in a separate json file 
## Example of a downloaded json
## {"batchcomplete":"","query":{"normalized":[{"from":"O'Connor_Peak","to":"O'Connor Peak"}],"pages":{"18547972":{"pageid":18547972,"ns":0,"title":"O'Connor Peak","extract":"O'Connor Peak (54\u00b016\u2032S 36\u00b019\u2032W) is a mountain peak, 675 m, standing west of Long Point on Barff Peninsula, South Georgia. Charted by a Norwegian Antarctic Expedition, 1927\u201328, and named Mount Bryde. Recharted by DI in 1929 and named after Midshipman W. P. O'Connor, Royal Navy Reserve, who assisted with the survey.\n This article incorporates public domain material from \"O'Connor Peak\". Geographic Names Information System. United States Geological Survey."}}}}
