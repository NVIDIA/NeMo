#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# =============================================================================
# Copyright 2017--2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# =============================================================================

"""
SacreBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores.
Inspired by Rico Sennrich's `multi-bleu-detok.perl`, it produces the official WMT scores but works with plain text.
It also knows all the standard test sets and handles downloading, processing, and tokenization for you.

See the [README.md] file for more information.
"""

import argparse
import gzip
import hashlib
import io
import math
import os
import re
import ssl
import sys
import unicodedata
import urllib.request
from collections import Counter, namedtuple
from itertools import zip_longest
from typing import Iterable, List, Tuple, Union

from nemo import logging
from nemo.collections.nlp.data.tokenizers.fairseq_tokenizer import tokenize_en

VERSION = '1.3.5'

try:
    # SIGPIPE is not available on Windows machines, throwing an exception.
    from signal import SIGPIPE

    # If SIGPIPE is available, change behaviour to default instead of ignore.
    from signal import signal, SIG_DFL

    signal(SIGPIPE, SIG_DFL)

except ImportError:
    logging.warning('Could not import signal.SIGPIPE (this is expected on Windows machines)')

# Where to store downloaded test sets.
# Define the environment variable $SACREBLEU, or use the default of ~/.sacrebleu
#
# Querying for a HOME environment variable can result in None (e.g., on Windows)
# in which case the os.path.join() throws a TypeError. Using expanduser() is
# a safe way to get the user's home folder.
USERHOME = os.path.expanduser("~")
SACREBLEU_DIR = os.environ.get('SACREBLEU', os.path.join(USERHOME, '.sacrebleu'))

# n-gram order. Don't change this.
NGRAM_ORDER = 4

# Default values for CHRF
CHRF_ORDER = 6
# default to 2 (per http://www.aclweb.org/anthology/W16-2341)
CHRF_BETA = 2

# The default floor value to use with `--smooth floor`
SMOOTH_VALUE_DEFAULT = 0.0

# This defines data locations. At the top level are test sets. Beneath each
# test set, we define the location to download the test data. The other keys
# are each language pair contained in the tarball, and the respective
# locations of the source and reference data within each. Many of these are
# *.sgm files, which are processed to produced plain text that can be used by
# this script. The canonical location of unpacked, processed data is
# $SACREBLEU_DIR/$TEST/$SOURCE-$TARGET.{$SOURCE,$TARGET}

# pep8: disable=E501
DATASETS = {
    'mtnt2019': {
        'data': ['http://www.cs.cmu.edu/~pmichel1/hosting/MTNT2019.tar.gz'],
        'description': 'Test set for the WMT 19 robustness shared task',
        'md5': ['78a672e1931f106a8549023c0e8af8f6'],
        'en-fr': ['2:MTNT2019/en-fr.final.tsv', '3:MTNT2019/en-fr.final.tsv'],
        'fr-en': ['2:MTNT2019/fr-en.final.tsv', '3:MTNT2019/fr-en.final.tsv'],
        'en-ja': ['2:MTNT2019/en-ja.final.tsv', '3:MTNT2019/en-ja.final.tsv'],
        'ja-en': ['2:MTNT2019/ja-en.final.tsv', '3:MTNT2019/ja-en.final.tsv'],
    },
    'mtnt1.1/test': {
        'data': ['https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz'],
        'description': 'Test data for the Machine Translation of Noisy Text task: '
        'http://www.cs.cmu.edu/~pmichel1/mtnt/',
        'citation': '@InProceedings{michel2018a:mtnt,\n    author = "Michel, Paul and Neubig, Graham",\n    title = '
        '"MTNT: A Testbed for Machine Translation of Noisy Text",\n    booktitle = "Proceedings of the '
        '2018 Conference on Empirical Methods in Natural Language Processing",\n    year = "2018",'
        '\n    publisher = "Association for Computational Linguistics",\n    pages = "543--553",'
        '\n    location = "Brussels, Belgium",\n    url = "http://aclweb.org/anthology/D18-1050"\n}',
        'md5': ['8ce1831ac584979ba8cdcd9d4be43e1d'],
        'en-fr': ['1:MTNT/test/test.en-fr.tsv', '2:MTNT/test/test.en-fr.tsv'],
        'fr-en': ['1:MTNT/test/test.fr-en.tsv', '2:MTNT/test/test.fr-en.tsv'],
        'en-ja': ['1:MTNT/test/test.en-ja.tsv', '2:MTNT/test/test.en-ja.tsv'],
        'ja-en': ['1:MTNT/test/test.ja-en.tsv', '2:MTNT/test/test.ja-en.tsv'],
    },
    'mtnt1.1/valid': {
        'data': ['https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz'],
        'description': 'Validation data for the Machine Translation of Noisy Text task: '
        'http://www.cs.cmu.edu/~pmichel1/mtnt/',
        'citation': '@InProceedings{michel2018a:mtnt,\n    author = "Michel, Paul and Neubig, Graham",\n    title = '
        '"MTNT: A Testbed for Machine Translation of Noisy Text",\n    booktitle = "Proceedings of the '
        '2018 Conference on Empirical Methods in Natural Language Processing",\n    year = "2018",'
        '\n    publisher = "Association for Computational Linguistics",\n    pages = "543--553",'
        '\n    location = "Brussels, Belgium",\n    url = "http://aclweb.org/anthology/D18-1050"\n}',
        'md5': ['8ce1831ac584979ba8cdcd9d4be43e1d'],
        'en-fr': ['1:MTNT/valid/valid.en-fr.tsv', '2:MTNT/valid/valid.en-fr.tsv'],
        'fr-en': ['1:MTNT/valid/valid.fr-en.tsv', '2:MTNT/valid/valid.fr-en.tsv'],
        'en-ja': ['1:MTNT/valid/valid.en-ja.tsv', '2:MTNT/valid/valid.en-ja.tsv'],
        'ja-en': ['1:MTNT/valid/valid.ja-en.tsv', '2:MTNT/valid/valid.ja-en.tsv'],
    },
    'mtnt1.1/train': {
        'data': ['https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz'],
        'description': 'Training data for the Machine Translation of Noisy Text task: '
        'http://www.cs.cmu.edu/~pmichel1/mtnt/',
        'citation': '@InProceedings{michel2018a:mtnt,\n    author = "Michel, Paul and Neubig, Graham",\n    title = '
        '"MTNT: A Testbed for Machine Translation of Noisy Text",\n    booktitle = "Proceedings of the '
        '2018 Conference on Empirical Methods in Natural Language Processing",\n    year = "2018",'
        '\n    publisher = "Association for Computational Linguistics",\n    pages = "543--553",'
        '\n    location = "Brussels, Belgium",\n    url = "http://aclweb.org/anthology/D18-1050"\n}',
        'md5': ['8ce1831ac584979ba8cdcd9d4be43e1d'],
        'en-fr': ['1:MTNT/train/train.en-fr.tsv', '2:MTNT/train/train.en-fr.tsv'],
        'fr-en': ['1:MTNT/train/train.fr-en.tsv', '2:MTNT/train/train.fr-en.tsv'],
        'en-ja': ['1:MTNT/train/train.en-ja.tsv', '2:MTNT/train/train.en-ja.tsv'],
        'ja-en': ['1:MTNT/train/train.ja-en.tsv', '2:MTNT/train/train.ja-en.tsv'],
    },
    'wmt19': {
        'data': ['http://data.statmt.org/wmt19/translation-task/test.tgz'],
        'md5': ['84de7162d158e28403103b01aeefc39a'],
        'cs-de': ['sgm/newstest2019-csde-src.cs.sgm', 'sgm/newstest2019-csde-ref.de.sgm'],
        'de-cs': ['sgm/newstest2019-decs-src.de.sgm', 'sgm/newstest2019-decs-ref.cs.sgm'],
        'de-en': ['sgm/newstest2019-deen-src.de.sgm', 'sgm/newstest2019-deen-ref.en.sgm'],
        'de-fr': ['sgm/newstest2019-defr-src.de.sgm', 'sgm/newstest2019-defr-ref.fr.sgm'],
        'en-cs': ['sgm/newstest2019-encs-src.en.sgm', 'sgm/newstest2019-encs-ref.cs.sgm'],
        'en-de': ['sgm/newstest2019-ende-src.en.sgm', 'sgm/newstest2019-ende-ref.de.sgm'],
        'en-fi': ['sgm/newstest2019-enfi-src.en.sgm', 'sgm/newstest2019-enfi-ref.fi.sgm'],
        'en-gu': ['sgm/newstest2019-engu-src.en.sgm', 'sgm/newstest2019-engu-ref.gu.sgm'],
        'en-kk': ['sgm/newstest2019-enkk-src.en.sgm', 'sgm/newstest2019-enkk-ref.kk.sgm'],
        'en-lt': ['sgm/newstest2019-enlt-src.en.sgm', 'sgm/newstest2019-enlt-ref.lt.sgm'],
        'en-ru': ['sgm/newstest2019-enru-src.en.sgm', 'sgm/newstest2019-enru-ref.ru.sgm'],
        'en-zh': ['sgm/newstest2019-enzh-src.en.sgm', 'sgm/newstest2019-enzh-ref.zh.sgm'],
        'fi-en': ['sgm/newstest2019-fien-src.fi.sgm', 'sgm/newstest2019-fien-ref.en.sgm'],
        'fr-de': ['sgm/newstest2019-frde-src.fr.sgm', 'sgm/newstest2019-frde-ref.de.sgm'],
        'gu-en': ['sgm/newstest2019-guen-src.gu.sgm', 'sgm/newstest2019-guen-ref.en.sgm'],
        'kk-en': ['sgm/newstest2019-kken-src.kk.sgm', 'sgm/newstest2019-kken-ref.en.sgm'],
        'lt-en': ['sgm/newstest2019-lten-src.lt.sgm', 'sgm/newstest2019-lten-ref.en.sgm'],
        'ru-en': ['sgm/newstest2019-ruen-src.ru.sgm', 'sgm/newstest2019-ruen-ref.en.sgm'],
        'zh-en': ['sgm/newstest2019-zhen-src.zh.sgm', 'sgm/newstest2019-zhen-ref.en.sgm'],
    },
    'wmt19/dev': {
        'data': ['http://data.statmt.org/wmt19/translation-task/dev.tgz'],
        'description': 'Development data for tasks new to 2019.',
        'md5': ['f2ec7af5947c19e0cacb3882eb208002'],
        'lt-en': ['dev/newsdev2019-lten-src.lt.sgm', 'dev/newsdev2019-lten-ref.en.sgm'],
        'en-lt': ['dev/newsdev2019-enlt-src.en.sgm', 'dev/newsdev2019-enlt-ref.lt.sgm'],
        'gu-en': ['dev/newsdev2019-guen-src.gu.sgm', 'dev/newsdev2019-guen-ref.en.sgm'],
        'en-gu': ['dev/newsdev2019-engu-src.en.sgm', 'dev/newsdev2019-engu-ref.gu.sgm'],
        'kk-en': ['dev/newsdev2019-kken-src.kk.sgm', 'dev/newsdev2019-kken-ref.en.sgm'],
        'en-kk': ['dev/newsdev2019-enkk-src.en.sgm', 'dev/newsdev2019-enkk-ref.kk.sgm'],
    },
    'wmt18': {
        'data': ['http://data.statmt.org/wmt18/translation-task/test.tgz'],
        'md5': ['f996c245ecffea23d0006fa4c34e9064'],
        'description': 'Official evaluation data.',
        'citation': '@inproceedings{bojar-etal-2018-findings,\n    title = "Findings of the 2018 Conference on '
        'Machine Translation ({WMT}18)",\n    author = "Bojar, Ond{\v{r}}ej  and\n      Federmann, '
        'Christian  and\n      Fishel, Mark  and\n      Graham, Yvette  and\n      Haddow, Barry  and\n  '
        '    Koehn, Philipp  and\n      Monz, Christof",\n    booktitle = "Proceedings of the Third '
        'Conference on Machine Translation: Shared Task Papers",\n    month = oct,\n    year = "2018",'
        '\n    address = "Belgium, Brussels",\n    publisher = "Association for Computational '
        'Linguistics",\n    url = "https://www.aclweb.org/anthology/W18-6401",\n    pages = "272--303",'
        '\n}',
        'cs-en': ['test/newstest2018-csen-src.cs.sgm', 'test/newstest2018-csen-ref.en.sgm'],
        'de-en': ['test/newstest2018-deen-src.de.sgm', 'test/newstest2018-deen-ref.en.sgm'],
        'en-cs': ['test/newstest2018-encs-src.en.sgm', 'test/newstest2018-encs-ref.cs.sgm'],
        'en-de': ['test/newstest2018-ende-src.en.sgm', 'test/newstest2018-ende-ref.de.sgm'],
        'en-et': ['test/newstest2018-enet-src.en.sgm', 'test/newstest2018-enet-ref.et.sgm'],
        'en-fi': ['test/newstest2018-enfi-src.en.sgm', 'test/newstest2018-enfi-ref.fi.sgm'],
        'en-ru': ['test/newstest2018-enru-src.en.sgm', 'test/newstest2018-enru-ref.ru.sgm'],
        'et-en': ['test/newstest2018-eten-src.et.sgm', 'test/newstest2018-eten-ref.en.sgm'],
        'fi-en': ['test/newstest2018-fien-src.fi.sgm', 'test/newstest2018-fien-ref.en.sgm'],
        'ru-en': ['test/newstest2018-ruen-src.ru.sgm', 'test/newstest2018-ruen-ref.en.sgm'],
        'en-tr': ['test/newstest2018-entr-src.en.sgm', 'test/newstest2018-entr-ref.tr.sgm'],
        'tr-en': ['test/newstest2018-tren-src.tr.sgm', 'test/newstest2018-tren-ref.en.sgm'],
        'en-zh': ['test/newstest2018-enzh-src.en.sgm', 'test/newstest2018-enzh-ref.zh.sgm'],
        'zh-en': ['test/newstest2018-zhen-src.zh.sgm', 'test/newstest2018-zhen-ref.en.sgm'],
    },
    'wmt18/test-ts': {
        'data': ['http://data.statmt.org/wmt18/translation-task/test-ts.tgz'],
        'md5': ['5c621a34d512cc2dd74162ae7d00b320'],
        'description': 'Official evaluation sources with extra test sets interleaved.',
        'cs-en': ['test/newstest2018-csen-src-ts.cs.sgm'],
        'de-en': ['test/newstest2018-deen-src-ts.de.sgm'],
        'en-cs': ['test/newstest2018-encs-src-ts.en.sgm'],
        'en-de': ['test/newstest2018-ende-src-ts.en.sgm'],
        'en-et': ['test/newstest2018-enet-src-ts.en.sgm'],
        'en-fi': ['test/newstest2018-enfi-src-ts.en.sgm'],
        'en-ru': ['test/newstest2018-enru-src-ts.en.sgm'],
        'et-en': ['test/newstest2018-eten-src-ts.et.sgm'],
        'fi-en': ['test/newstest2018-fien-src-ts.fi.sgm'],
        'ru-en': ['test/newstest2018-ruen-src-ts.ru.sgm'],
        'en-tr': ['test/newstest2018-entr-src-ts.en.sgm'],
        'tr-en': ['test/newstest2018-tren-src-ts.tr.sgm'],
        'en-zh': ['test/newstest2018-enzh-src-ts.en.sgm'],
        'zh-en': ['test/newstest2018-zhen-src-ts.zh.sgm'],
    },
    'wmt18/dev': {
        'data': ['http://data.statmt.org/wmt18/translation-task/dev.tgz'],
        'md5': ['486f391da54a7a3247f02ebd25996f24'],
        'description': 'Development data (Estonian<>English).',
        'et-en': ['dev/newsdev2018-eten-src.et.sgm', 'dev/newsdev2018-eten-ref.en.sgm'],
        'en-et': ['dev/newsdev2018-enet-src.en.sgm', 'dev/newsdev2018-enet-ref.et.sgm'],
    },
    'wmt17': {
        'data': ['http://data.statmt.org/wmt17/translation-task/test.tgz'],
        'md5': ['86a1724c276004aa25455ae2a04cef26'],
        'description': 'Official evaluation data.',
        'citation': '@InProceedings{bojar-EtAl:2017:WMT1,\n  author    = {Bojar, Ond\\v{r}ej  and  Chatterjee, '
        'Rajen  and  Federmann, Christian  and  Graham, Yvette  and  Haddow, Barry  and  Huang, '
        'Shujian  and  Huck, Matthias  and  Koehn, Philipp  and  Liu, Qun  and  Logacheva, Varvara  and  '
        'Monz, Christof  and  Negri, Matteo  and  Post, Matt  and  Rubino, Raphael  and  Specia, '
        'Lucia  and  Turchi, Marco},\n  title     = {Findings of the 2017 Conference on Machine '
        'Translation (WMT17)},\n  booktitle = {Proceedings of the Second Conference on Machine '
        'Translation, Volume 2: Shared Task Papers},\n  month     = {September},\n  year      = {2017},'
        '\n  address   = {Copenhagen, Denmark},\n  publisher = {Association for Computational '
        'Linguistics},\n  pages     = {169--214},\n  url       = {'
        'http://www.aclweb.org/anthology/W17-4717}\n}',
        'cs-en': ['test/newstest2017-csen-src.cs.sgm', 'test/newstest2017-csen-ref.en.sgm'],
        'de-en': ['test/newstest2017-deen-src.de.sgm', 'test/newstest2017-deen-ref.en.sgm'],
        'en-cs': ['test/newstest2017-encs-src.en.sgm', 'test/newstest2017-encs-ref.cs.sgm'],
        'en-de': ['test/newstest2017-ende-src.en.sgm', 'test/newstest2017-ende-ref.de.sgm'],
        'en-fi': ['test/newstest2017-enfi-src.en.sgm', 'test/newstest2017-enfi-ref.fi.sgm'],
        'en-lv': ['test/newstest2017-enlv-src.en.sgm', 'test/newstest2017-enlv-ref.lv.sgm'],
        'en-ru': ['test/newstest2017-enru-src.en.sgm', 'test/newstest2017-enru-ref.ru.sgm'],
        'en-tr': ['test/newstest2017-entr-src.en.sgm', 'test/newstest2017-entr-ref.tr.sgm'],
        'en-zh': ['test/newstest2017-enzh-src.en.sgm', 'test/newstest2017-enzh-ref.zh.sgm'],
        'fi-en': ['test/newstest2017-fien-src.fi.sgm', 'test/newstest2017-fien-ref.en.sgm'],
        'lv-en': ['test/newstest2017-lven-src.lv.sgm', 'test/newstest2017-lven-ref.en.sgm'],
        'ru-en': ['test/newstest2017-ruen-src.ru.sgm', 'test/newstest2017-ruen-ref.en.sgm'],
        'tr-en': ['test/newstest2017-tren-src.tr.sgm', 'test/newstest2017-tren-ref.en.sgm'],
        'zh-en': ['test/newstest2017-zhen-src.zh.sgm', 'test/newstest2017-zhen-ref.en.sgm'],
    },
    'wmt17/B': {
        'data': ['http://data.statmt.org/wmt17/translation-task/test.tgz'],
        'md5': ['86a1724c276004aa25455ae2a04cef26'],
        'description': 'Additional reference for EN-FI and FI-EN.',
        'en-fi': ['test/newstestB2017-enfi-src.en.sgm', 'test/newstestB2017-enfi-ref.fi.sgm'],
    },
    'wmt17/tworefs': {
        'data': ['http://data.statmt.org/wmt17/translation-task/test.tgz'],
        'md5': ['86a1724c276004aa25455ae2a04cef26'],
        'description': 'Systems with two references.',
        'en-fi': [
            'test/newstest2017-enfi-src.en.sgm',
            'test/newstest2017-enfi-ref.fi.sgm',
            'test/newstestB2017-enfi-ref.fi.sgm',
        ],
    },
    'wmt17/improved': {
        'data': ['http://data.statmt.org/wmt17/translation-task/test-update-1.tgz'],
        'md5': ['91dbfd5af99bc6891a637a68e04dfd41'],
        'description': 'Improved zh-en and en-zh translations.',
        'en-zh': ['newstest2017-enzh-src.en.sgm', 'newstest2017-enzh-ref.zh.sgm'],
        'zh-en': ['newstest2017-zhen-src.zh.sgm', 'newstest2017-zhen-ref.en.sgm'],
    },
    'wmt17/dev': {
        'data': ['http://data.statmt.org/wmt17/translation-task/dev.tgz'],
        'md5': ['9b1aa63c1cf49dccdd20b962fe313989'],
        'description': 'Development sets released for new languages in 2017.',
        'en-lv': ['dev/newsdev2017-enlv-src.en.sgm', 'dev/newsdev2017-enlv-ref.lv.sgm'],
        'en-zh': ['dev/newsdev2017-enzh-src.en.sgm', 'dev/newsdev2017-enzh-ref.zh.sgm'],
        'lv-en': ['dev/newsdev2017-lven-src.lv.sgm', 'dev/newsdev2017-lven-ref.en.sgm'],
        'zh-en': ['dev/newsdev2017-zhen-src.zh.sgm', 'dev/newsdev2017-zhen-ref.en.sgm'],
    },
    'wmt17/ms': {
        'data': [
            'https://github.com/MicrosoftTranslator/Translator-HumanParityData/archive/master.zip',
            'http://data.statmt.org/wmt17/translation-task/test-update-1.tgz',
        ],
        'md5': ['18fdaa7a3c84cf6ef688da1f6a5fa96f', '91dbfd5af99bc6891a637a68e04dfd41'],
        'description': 'Additional Chinese-English references from Microsoft Research.',
        'citation': '@inproceedings{achieving-human-parity-on-automatic-chinese-to-english-news-translation,'
        '\n  author = {Hassan Awadalla, Hany and Aue, Anthony and Chen, Chang and Chowdhary, Vishal and '
        'Clark, Jonathan and Federmann, Christian and Huang, Xuedong and Junczys-Dowmunt, Marcin and '
        'Lewis, Will and Li, Mu and Liu, Shujie and Liu, Tie-Yan and Luo, Renqian and Menezes, '
        'Arul and Qin, Tao and Seide, Frank and Tan, Xu and Tian, Fei and Wu, Lijun and Wu, '
        'Shuangzhi and Xia, Yingce and Zhang, Dongdong and Zhang, Zhirui and Zhou, Ming},\n  title = {'
        'Achieving Human Parity on Automatic Chinese to English News Translation},\n  booktitle = {},'
        '\n  year = {2018},\n  month = {March},\n  abstract = {Machine translation has made rapid '
        'advances in recent years. Millions of people are using it today in online translation systems '
        'and mobile applications in order to communicate across language barriers. The question '
        'naturally arises whether such systems can approach or achieve parity with human translations. '
        'In this paper, we first address the problem of how to define and accurately measure human '
        'parity in translation. We then describe Microsoft’s machine translation system and measure the '
        'quality of its translations on the widely used WMT 2017 news translation task from Chinese to '
        'English. We find that our latest neural machine translation system has reached a new '
        'state-of-the-art, and that the translation quality is at human parity when compared to '
        'professional human translations. We also find that it significantly exceeds the quality of '
        'crowd-sourced non-professional translations.},\n  publisher = {},\n  url = {'
        'https://www.microsoft.com/en-us/research/publication/achieving-human-parity-on-automatic'
        '-chinese-to-english-news-translation/},\n  address = {},\n  pages = {},\n  journal = {},'
        '\n  volume = {},\n  chapter = {},\n  isbn = {},\n}',
        'zh-en': [
            'newstest2017-zhen-src.zh.sgm',
            'newstest2017-zhen-ref.en.sgm',
            'Translator-HumanParityData-master/Translator-HumanParityData/References/Translator-HumanParityData'
            + '-Reference-HT.txt',
            'Translator-HumanParityData-master/Translator-HumanParityData/References/Translator-HumanParityData'
            + '-Reference-PE.txt',
        ],
    },
    'wmt16': {
        'data': ['http://data.statmt.org/wmt16/translation-task/test.tgz'],
        'md5': ['3d809cd0c2c86adb2c67034d15c4e446'],
        'description': 'Official evaluation data.',
        'citation': '@InProceedings{bojar-EtAl:2016:WMT1,\n  author    = {Bojar, Ond\\v{r}ej  and  Chatterjee, '
        'Rajen  and  Federmann, Christian  and  Graham, Yvette  and  Haddow, Barry  and  Huck, '
        'Matthias  and  Jimeno Yepes, Antonio  and  Koehn, Philipp  and  Logacheva, Varvara  and  Monz, '
        'Christof  and  Negri, Matteo  and  Neveol, Aurelie  and  Neves, Mariana  and  Popel, '
        'Martin  and  Post, Matt  and  Rubino, Raphael  and  Scarton, Carolina  and  Specia, Lucia  and  '
        'Turchi, Marco  and  Verspoor, Karin  and  Zampieri, Marcos},\n  title     = {Findings of the '
        '2016 Conference on Machine Translation},\n  booktitle = {Proceedings of the First Conference on '
        'Machine Translation},\n  month     = {August},\n  year      = {2016},\n  address   = {Berlin, '
        'Germany},\n  publisher = {Association for Computational Linguistics},\n  pages     = {'
        '131--198},\n  url       = {http://www.aclweb.org/anthology/W/W16/W16-2301}\n}',
        'cs-en': ['test/newstest2016-csen-src.cs.sgm', 'test/newstest2016-csen-ref.en.sgm'],
        'de-en': ['test/newstest2016-deen-src.de.sgm', 'test/newstest2016-deen-ref.en.sgm'],
        'en-cs': ['test/newstest2016-encs-src.en.sgm', 'test/newstest2016-encs-ref.cs.sgm'],
        'en-de': ['test/newstest2016-ende-src.en.sgm', 'test/newstest2016-ende-ref.de.sgm'],
        'en-fi': ['test/newstest2016-enfi-src.en.sgm', 'test/newstest2016-enfi-ref.fi.sgm'],
        'en-ro': ['test/newstest2016-enro-src.en.sgm', 'test/newstest2016-enro-ref.ro.sgm'],
        'en-ru': ['test/newstest2016-enru-src.en.sgm', 'test/newstest2016-enru-ref.ru.sgm'],
        'en-tr': ['test/newstest2016-entr-src.en.sgm', 'test/newstest2016-entr-ref.tr.sgm'],
        'fi-en': ['test/newstest2016-fien-src.fi.sgm', 'test/newstest2016-fien-ref.en.sgm'],
        'ro-en': ['test/newstest2016-roen-src.ro.sgm', 'test/newstest2016-roen-ref.en.sgm'],
        'ru-en': ['test/newstest2016-ruen-src.ru.sgm', 'test/newstest2016-ruen-ref.en.sgm'],
        'tr-en': ['test/newstest2016-tren-src.tr.sgm', 'test/newstest2016-tren-ref.en.sgm'],
    },
    'wmt16/B': {
        'data': ['http://data.statmt.org/wmt16/translation-task/test.tgz'],
        'md5': ['3d809cd0c2c86adb2c67034d15c4e446'],
        'description': 'Additional reference for EN-FI.',
        'en-fi': ['test/newstest2016-enfi-src.en.sgm', 'test/newstestB2016-enfi-ref.fi.sgm'],
    },
    'wmt16/tworefs': {
        'data': ['http://data.statmt.org/wmt16/translation-task/test.tgz'],
        'md5': ['3d809cd0c2c86adb2c67034d15c4e446'],
        'description': 'EN-FI with two references.',
        'en-fi': [
            'test/newstest2016-enfi-src.en.sgm',
            'test/newstest2016-enfi-ref.fi.sgm',
            'test/newstestB2016-enfi-ref.fi.sgm',
        ],
    },
    'wmt16/dev': {
        'data': ['http://data.statmt.org/wmt16/translation-task/dev.tgz'],
        'md5': ['4a3dc2760bb077f4308cce96b06e6af6'],
        'description': 'Development sets released for new languages in 2016.',
        'en-ro': ['dev/newsdev2016-enro-src.en.sgm', 'dev/newsdev2016-enro-ref.ro.sgm'],
        'en-tr': ['dev/newsdev2016-entr-src.en.sgm', 'dev/newsdev2016-entr-ref.tr.sgm'],
        'ro-en': ['dev/newsdev2016-roen-src.ro.sgm', 'dev/newsdev2016-roen-ref.en.sgm'],
        'tr-en': ['dev/newsdev2016-tren-src.tr.sgm', 'dev/newsdev2016-tren-ref.en.sgm'],
    },
    'wmt15': {
        'data': ['http://statmt.org/wmt15/test.tgz'],
        'md5': ['67e3beca15e69fe3d36de149da0a96df'],
        'description': 'Official evaluation data.',
        'citation': '@InProceedings{bojar-EtAl:2015:WMT,\n  author    = {Bojar, Ond\\v{r}ej  and  Chatterjee, '
        'Rajen  and  Federmann, Christian  and  Haddow, Barry  and  Huck, Matthias  and  Hokamp, '
        'Chris  and  Koehn, Philipp  and  Logacheva, Varvara  and  Monz, Christof  and  Negri, '
        'Matteo  and  Post, Matt  and  Scarton, Carolina  and  Specia, Lucia  and  Turchi, Marco},'
        '\n  title     = {Findings of the 2015 Workshop on Statistical Machine Translation},'
        '\n  booktitle = {Proceedings of the Tenth Workshop on Statistical Machine Translation},'
        '\n  month     = {September},\n  year      = {2015},\n  address   = {Lisbon, Portugal},'
        '\n  publisher = {Association for Computational Linguistics},\n  pages     = {1--46},\n  url     '
        '  = {http://aclweb.org/anthology/W15-3001}\n}',
        'en-fr': ['test/newsdiscusstest2015-enfr-src.en.sgm', 'test/newsdiscusstest2015-enfr-ref.fr.sgm'],
        'fr-en': ['test/newsdiscusstest2015-fren-src.fr.sgm', 'test/newsdiscusstest2015-fren-ref.en.sgm'],
        'cs-en': ['test/newstest2015-csen-src.cs.sgm', 'test/newstest2015-csen-ref.en.sgm'],
        'de-en': ['test/newstest2015-deen-src.de.sgm', 'test/newstest2015-deen-ref.en.sgm'],
        'en-cs': ['test/newstest2015-encs-src.en.sgm', 'test/newstest2015-encs-ref.cs.sgm'],
        'en-de': ['test/newstest2015-ende-src.en.sgm', 'test/newstest2015-ende-ref.de.sgm'],
        'en-fi': ['test/newstest2015-enfi-src.en.sgm', 'test/newstest2015-enfi-ref.fi.sgm'],
        'en-ru': ['test/newstest2015-enru-src.en.sgm', 'test/newstest2015-enru-ref.ru.sgm'],
        'fi-en': ['test/newstest2015-fien-src.fi.sgm', 'test/newstest2015-fien-ref.en.sgm'],
        'ru-en': ['test/newstest2015-ruen-src.ru.sgm', 'test/newstest2015-ruen-ref.en.sgm'],
    },
    'wmt14': {
        'data': ['http://statmt.org/wmt14/test-filtered.tgz'],
        'md5': ['84c597844c1542e29c2aff23aaee4310'],
        'description': 'Official evaluation data.',
        'citation': '@InProceedings{bojar-EtAl:2014:W14-33,\n  author    = {Bojar, Ondrej  and  Buck, Christian  and '
        ' Federmann, Christian  and  Haddow, Barry  and  Koehn, Philipp  and  Leveling, Johannes  and  '
        'Monz, Christof  and  Pecina, Pavel  and  Post, Matt  and  Saint-Amand, Herve  and  Soricut, '
        'Radu  and  Specia, Lucia  and  Tamchyna, Ale\\v{s}},\n  title     = {Findings of the 2014 '
        'Workshop on Statistical Machine Translation},\n  booktitle = {Proceedings of the Ninth Workshop '
        'on Statistical Machine Translation},\n  month     = {June},\n  year      = {2014},\n  address   '
        '= {Baltimore, Maryland, USA},\n  publisher = {Association for Computational Linguistics},'
        '\n  pages     = {12--58},\n  url       = {http://www.aclweb.org/anthology/W/W14/W14-3302}\n}',
        'cs-en': ['test/newstest2014-csen-src.cs.sgm', 'test/newstest2014-csen-ref.en.sgm'],
        'en-cs': ['test/newstest2014-csen-src.en.sgm', 'test/newstest2014-csen-ref.cs.sgm'],
        'de-en': ['test/newstest2014-deen-src.de.sgm', 'test/newstest2014-deen-ref.en.sgm'],
        'en-de': ['test/newstest2014-deen-src.en.sgm', 'test/newstest2014-deen-ref.de.sgm'],
        'en-fr': ['test/newstest2014-fren-src.en.sgm', 'test/newstest2014-fren-ref.fr.sgm'],
        'fr-en': ['test/newstest2014-fren-src.fr.sgm', 'test/newstest2014-fren-ref.en.sgm'],
        'en-hi': ['test/newstest2014-hien-src.en.sgm', 'test/newstest2014-hien-ref.hi.sgm'],
        'hi-en': ['test/newstest2014-hien-src.hi.sgm', 'test/newstest2014-hien-ref.en.sgm'],
        'en-ru': ['test/newstest2014-ruen-src.en.sgm', 'test/newstest2014-ruen-ref.ru.sgm'],
        'ru-en': ['test/newstest2014-ruen-src.ru.sgm', 'test/newstest2014-ruen-ref.en.sgm'],
    },
    'wmt14/full': {
        'data': ['http://statmt.org/wmt14/test-full.tgz'],
        'md5': ['a8cd784e006feb32ac6f3d9ec7eb389a'],
        'description': 'Evaluation data released after official evaluation for further research.',
        'cs-en': ['test-full/newstest2014-csen-src.cs.sgm', 'test-full/newstest2014-csen-ref.en.sgm'],
        'en-cs': ['test-full/newstest2014-csen-src.en.sgm', 'test-full/newstest2014-csen-ref.cs.sgm'],
        'de-en': ['test-full/newstest2014-deen-src.de.sgm', 'test-full/newstest2014-deen-ref.en.sgm'],
        'en-de': ['test-full/newstest2014-deen-src.en.sgm', 'test-full/newstest2014-deen-ref.de.sgm'],
        'en-fr': ['test-full/newstest2014-fren-src.en.sgm', 'test-full/newstest2014-fren-ref.fr.sgm'],
        'fr-en': ['test-full/newstest2014-fren-src.fr.sgm', 'test-full/newstest2014-fren-ref.en.sgm'],
        'en-hi': ['test-full/newstest2014-hien-src.en.sgm', 'test-full/newstest2014-hien-ref.hi.sgm'],
        'hi-en': ['test-full/newstest2014-hien-src.hi.sgm', 'test-full/newstest2014-hien-ref.en.sgm'],
        'en-ru': ['test-full/newstest2014-ruen-src.en.sgm', 'test-full/newstest2014-ruen-ref.ru.sgm'],
        'ru-en': ['test-full/newstest2014-ruen-src.ru.sgm', 'test-full/newstest2014-ruen-ref.en.sgm'],
    },
    'wmt13': {
        'data': ['http://statmt.org/wmt13/test.tgz'],
        'md5': ['48eca5d02f637af44e85186847141f67'],
        'description': 'Official evaluation data.',
        'citation': '@InProceedings{bojar-EtAl:2013:WMT,\n  author    = {Bojar, Ond\\v{r}ej  and  Buck, Christian  '
        'and  Callison-Burch, Chris  and  Federmann, Christian  and  Haddow, Barry  and  Koehn, '
        'Philipp  and  Monz, Christof  and  Post, Matt  and  Soricut, Radu  and  Specia, Lucia},'
        '\n  title     = {Findings of the 2013 {Workshop on Statistical Machine Translation}},'
        '\n  booktitle = {Proceedings of the Eighth Workshop on Statistical Machine Translation},'
        '\n  month     = {August},\n  year      = {2013},\n  address   = {Sofia, Bulgaria},\n  publisher '
        '= {Association for Computational Linguistics},\n  pages     = {1--44},\n  url       = {'
        'http://www.aclweb.org/anthology/W13-2201}\n}',
        'cs-en': ['test/newstest2013-src.cs.sgm', 'test/newstest2013-src.en.sgm'],
        'en-cs': ['test/newstest2013-src.en.sgm', 'test/newstest2013-src.cs.sgm'],
        'de-en': ['test/newstest2013-src.de.sgm', 'test/newstest2013-src.en.sgm'],
        'en-de': ['test/newstest2013-src.en.sgm', 'test/newstest2013-src.de.sgm'],
        'es-en': ['test/newstest2013-src.es.sgm', 'test/newstest2013-src.en.sgm'],
        'en-es': ['test/newstest2013-src.en.sgm', 'test/newstest2013-src.es.sgm'],
        'fr-en': ['test/newstest2013-src.fr.sgm', 'test/newstest2013-src.en.sgm'],
        'en-fr': ['test/newstest2013-src.en.sgm', 'test/newstest2013-src.fr.sgm'],
        'ru-en': ['test/newstest2013-src.ru.sgm', 'test/newstest2013-src.en.sgm'],
        'en-ru': ['test/newstest2013-src.en.sgm', 'test/newstest2013-src.ru.sgm'],
    },
    'wmt12': {
        'data': ['http://statmt.org/wmt12/test.tgz'],
        'md5': ['608232d34ebc4ba2ff70fead45674e47'],
        'description': 'Official evaluation data.',
        'citation': '@InProceedings{callisonburch-EtAl:2012:WMT,\n  author    = {Callison-Burch, Chris  and  Koehn, '
        'Philipp  and  Monz, Christof  and  Post, Matt  and  Soricut, Radu  and  Specia, Lucia},'
        '\n  title     = {Findings of the 2012 Workshop on Statistical Machine Translation},'
        '\n  booktitle = {Proceedings of the Seventh Workshop on Statistical Machine Translation},'
        '\n  month     = {June},\n  year      = {2012},\n  address   = {Montr{\'e}al, Canada},'
        '\n  publisher = {Association for Computational Linguistics},\n  pages     = {10--51},'
        '\n  url       = {http://www.aclweb.org/anthology/W12-3102}\n}',
        'cs-en': ['test/newstest2012-src.cs.sgm', 'test/newstest2012-src.en.sgm'],
        'en-cs': ['test/newstest2012-src.en.sgm', 'test/newstest2012-src.cs.sgm'],
        'de-en': ['test/newstest2012-src.de.sgm', 'test/newstest2012-src.en.sgm'],
        'en-de': ['test/newstest2012-src.en.sgm', 'test/newstest2012-src.de.sgm'],
        'es-en': ['test/newstest2012-src.es.sgm', 'test/newstest2012-src.en.sgm'],
        'en-es': ['test/newstest2012-src.en.sgm', 'test/newstest2012-src.es.sgm'],
        'fr-en': ['test/newstest2012-src.fr.sgm', 'test/newstest2012-src.en.sgm'],
        'en-fr': ['test/newstest2012-src.en.sgm', 'test/newstest2012-src.fr.sgm'],
    },
    'wmt11': {
        'data': ['http://statmt.org/wmt11/test.tgz'],
        'md5': ['b0c9680adf32d394aefc2b24e3a5937e'],
        'description': 'Official evaluation data.',
        'citation': '@InProceedings{callisonburch-EtAl:2011:WMT,\n  author    = {Callison-Burch, Chris  and  Koehn, '
        'Philipp  and  Monz, Christof  and  Zaidan, Omar},\n  title     = {Findings of the 2011 Workshop '
        'on Statistical Machine Translation},\n  booktitle = {Proceedings of the Sixth Workshop on '
        'Statistical Machine Translation},\n  month     = {July},\n  year      = {2011},\n  address   = '
        '{Edinburgh, Scotland},\n  publisher = {Association for Computational Linguistics},\n  pages     '
        '= {22--64},\n  url       = {http://www.aclweb.org/anthology/W11-2103}\n}',
        'cs-en': ['newstest2011-src.cs.sgm', 'newstest2011-src.en.sgm'],
        'en-cs': ['newstest2011-src.en.sgm', 'newstest2011-src.cs.sgm'],
        'de-en': ['newstest2011-src.de.sgm', 'newstest2011-src.en.sgm'],
        'en-de': ['newstest2011-src.en.sgm', 'newstest2011-src.de.sgm'],
        'fr-en': ['newstest2011-src.fr.sgm', 'newstest2011-src.en.sgm'],
        'en-fr': ['newstest2011-src.en.sgm', 'newstest2011-src.fr.sgm'],
        'es-en': ['newstest2011-src.es.sgm', 'newstest2011-src.en.sgm'],
        'en-es': ['newstest2011-src.en.sgm', 'newstest2011-src.es.sgm'],
    },
    'wmt10': {
        'data': ['http://statmt.org/wmt10/test.tgz'],
        'md5': ['491cb885a355da5a23ea66e7b3024d5c'],
        'description': 'Official evaluation data.',
        'citation': '@InProceedings{callisonburch-EtAl:2010:WMT,\n  author    = {Callison-Burch, Chris  and  Koehn, '
        'Philipp  and  Monz, Christof  and  Peterson, Kay  and  Przybocki, Mark  and  Zaidan, Omar},'
        '\n  title     = {Findings of the 2010 Joint Workshop on Statistical Machine Translation and '
        'Metrics for Machine Translation},\n  booktitle = {Proceedings of the Joint Fifth Workshop on '
        'Statistical Machine Translation and MetricsMATR},\n  month     = {July},\n  year      = {2010},'
        '\n  address   = {Uppsala, Sweden},\n  publisher = {Association for Computational Linguistics},'
        '\n  pages     = {17--53},\n  note      = {Revised August 2010},\n  url       = {'
        'http://www.aclweb.org/anthology/W10-1703}\n}',
        'cs-en': ['test/newstest2010-src.cz.sgm', 'test/newstest2010-src.en.sgm'],
        'en-cs': ['test/newstest2010-src.en.sgm', 'test/newstest2010-src.cz.sgm'],
        'de-en': ['test/newstest2010-src.de.sgm', 'test/newstest2010-src.en.sgm'],
        'en-de': ['test/newstest2010-src.en.sgm', 'test/newstest2010-src.de.sgm'],
        'es-en': ['test/newstest2010-src.es.sgm', 'test/newstest2010-src.en.sgm'],
        'en-es': ['test/newstest2010-src.en.sgm', 'test/newstest2010-src.es.sgm'],
        'fr-en': ['test/newstest2010-src.fr.sgm', 'test/newstest2010-src.en.sgm'],
        'en-fr': ['test/newstest2010-src.en.sgm', 'test/newstest2010-src.fr.sgm'],
    },
    'wmt09': {
        'data': ['http://statmt.org/wmt09/test.tgz'],
        'md5': ['da227abfbd7b666ec175b742a0d27b37'],
        'description': 'Official evaluation data.',
        'citation': '@InProceedings{callisonburch-EtAl:2009:WMT-09,\n  author    = {Callison-Burch, Chris  and  '
        'Koehn, Philipp  and  Monz, Christof  and  Schroeder, Josh},\n  title     = {Findings of the '
        '2009 {W}orkshop on {S}tatistical {M}achine {T}ranslation},\n  booktitle = {Proceedings of the '
        'Fourth Workshop on Statistical Machine Translation},\n  month     = {March},\n  year      = {'
        '2009},\n  address   = {Athens, Greece},\n  publisher = {Association for Computational '
        'Linguistics},\n  pages     = {1--28},\n  url       = {'
        'http://www.aclweb.org/anthology/W/W09/W09-0401}\n}',
        'cs-en': ['test/newstest2009-src.cz.sgm', 'test/newstest2009-src.en.sgm'],
        'en-cs': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.cz.sgm'],
        'de-en': ['test/newstest2009-src.de.sgm', 'test/newstest2009-src.en.sgm'],
        'en-de': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.de.sgm'],
        'es-en': ['test/newstest2009-src.es.sgm', 'test/newstest2009-src.en.sgm'],
        'en-es': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.es.sgm'],
        'fr-en': ['test/newstest2009-src.fr.sgm', 'test/newstest2009-src.en.sgm'],
        'en-fr': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.fr.sgm'],
        'hu-en': ['test/newstest2009-src.hu.sgm', 'test/newstest2009-src.en.sgm'],
        'en-hu': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.hu.sgm'],
        'it-en': ['test/newstest2009-src.it.sgm', 'test/newstest2009-src.en.sgm'],
        'en-it': ['test/newstest2009-src.en.sgm', 'test/newstest2009-src.it.sgm'],
    },
    'wmt08': {
        'data': ['http://statmt.org/wmt08/test.tgz'],
        'md5': ['0582e4e894a3342044059c894e1aea3d'],
        'description': 'Official evaluation data.',
        'citation': '@InProceedings{callisonburch-EtAl:2008:WMT,\n  author    = {Callison-Burch, Chris  and  '
        'Fordyce, Cameron  and  Koehn, Philipp  and  Monz, Christof  and  Schroeder, Josh},\n  title     '
        '= {Further Meta-Evaluation of Machine Translation},\n  booktitle = {Proceedings of the Third '
        'Workshop on Statistical Machine Translation},\n  month     = {June},\n  year      = {2008},'
        '\n  address   = {Columbus, Ohio},\n  publisher = {Association for Computational Linguistics},'
        '\n  pages     = {70--106},\n  url       = {http://www.aclweb.org/anthology/W/W08/W08-0309}\n}',
        'cs-en': ['test/newstest2008-src.cz.sgm', 'test/newstest2008-src.en.sgm'],
        'en-cs': ['test/newstest2008-src.en.sgm', 'test/newstest2008-src.cz.sgm'],
        'de-en': ['test/newstest2008-src.de.sgm', 'test/newstest2008-src.en.sgm'],
        'en-de': ['test/newstest2008-src.en.sgm', 'test/newstest2008-src.de.sgm'],
        'es-en': ['test/newstest2008-src.es.sgm', 'test/newstest2008-src.en.sgm'],
        'en-es': ['test/newstest2008-src.en.sgm', 'test/newstest2008-src.es.sgm'],
        'fr-en': ['test/newstest2008-src.fr.sgm', 'test/newstest2008-src.en.sgm'],
        'en-fr': ['test/newstest2008-src.en.sgm', 'test/newstest2008-src.fr.sgm'],
        'hu-en': ['test/newstest2008-src.hu.sgm', 'test/newstest2008-src.en.sgm'],
        'en-hu': ['test/newstest2008-src.en.sgm', 'test/newstest2008-src.hu.sgm'],
    },
    'wmt08/nc': {
        'data': ['http://statmt.org/wmt08/test.tgz'],
        'md5': ['0582e4e894a3342044059c894e1aea3d'],
        'description': 'Official evaluation data (news commentary).',
        'cs-en': ['test/nc-test2008-src.cz.sgm', 'test/nc-test2008-src.en.sgm'],
        'en-cs': ['test/nc-test2008-src.en.sgm', 'test/nc-test2008-src.cz.sgm'],
    },
    'wmt08/europarl': {
        'data': ['http://statmt.org/wmt08/test.tgz'],
        'md5': ['0582e4e894a3342044059c894e1aea3d'],
        'description': 'Official evaluation data (Europarl).',
        'de-en': ['test/test2008-src.de.sgm', 'test/test2008-src.en.sgm'],
        'en-de': ['test/test2008-src.en.sgm', 'test/test2008-src.de.sgm'],
        'es-en': ['test/test2008-src.es.sgm', 'test/test2008-src.en.sgm'],
        'en-es': ['test/test2008-src.en.sgm', 'test/test2008-src.es.sgm'],
        'fr-en': ['test/test2008-src.fr.sgm', 'test/test2008-src.en.sgm'],
        'en-fr': ['test/test2008-src.en.sgm', 'test/test2008-src.fr.sgm'],
    },
    'iwslt17': {
        'data': [
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/en/fr/en-fr.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/fr/en/fr-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/en/de/en-de.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/de/en/de-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/en/ar/en-ar.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/ar/en/ar-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/en/ja/en-ja.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/ja/en/ja-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/en/ko/en-ko.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/ko/en/ko-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/en/zh/en-zh.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/zh/en/zh-en.tgz',
        ],
        'md5': [
            "1849bcc3b006dc0642a8843b11aa7192",
            "79bf7a2ef02d226875f55fb076e7e473",
            "b68e7097b179491f6c466ef41ad72b9b",
            "e3f5b2a075a2da1a395c8b60bf1e9be1",
            "ecdc6bc4ab4c8984e919444f3c05183a",
            "4b5141d14b98706c081371e2f8afe0ca",
            "d957ee79de1f33c89077d37c5a2c5b06",
            "c213e8bb918ebf843543fe9fd2e33db2",
            "59f6a81c707378176e9ad8bb8d811f5f",
            "7e580af973bb389ec1d1378a1850742f",
            "975a858783a0ebec8c57d83ddd5bd381",
            "cc51d9b7fe1ff2af858c6a0dd80b8815",
        ],
        'description': 'Official evaluation data for IWSLT.',
        'citation': '@InProceedings{iwslt2017,\n  author    = {Cettolo, Mauro and Federico, Marcello and Bentivogli, '
        'Luisa and Niehues, Jan and Stüker, Sebastian and Sudoh, Katsuitho and Yoshino, Koichiro and '
        'Federmann, Christian},\n  title     = {Overview of the IWSLT 2017 Evaluation Campaign},'
        '\n  booktitle = {14th International Workshop on Spoken Language Translation},\n  month     = {'
        'December},\n  year      = {2017},\n  address   = {Tokyo, Japan},\n  pages     = {2--14},'
        '\n  url       = {http://workshop2017.iwslt.org/downloads/iwslt2017_proceeding_v2.pdf}\n}',
        'en-fr': ['en-fr/IWSLT17.TED.tst2017.en-fr.en.xml', 'fr-en/IWSLT17.TED.tst2017.fr-en.fr.xml'],
        'fr-en': ['fr-en/IWSLT17.TED.tst2017.fr-en.fr.xml', 'en-fr/IWSLT17.TED.tst2017.en-fr.en.xml'],
        'en-de': ['en-de/IWSLT17.TED.tst2017.en-de.en.xml', 'de-en/IWSLT17.TED.tst2017.de-en.de.xml'],
        'de-en': ['de-en/IWSLT17.TED.tst2017.de-en.de.xml', 'en-de/IWSLT17.TED.tst2017.en-de.en.xml'],
        'en-zh': ['en-zh/IWSLT17.TED.tst2017.en-zh.en.xml', 'zh-en/IWSLT17.TED.tst2017.zh-en.zh.xml'],
        'zh-en': ['zh-en/IWSLT17.TED.tst2017.zh-en.zh.xml', 'en-zh/IWSLT17.TED.tst2017.en-zh.en.xml'],
    },
    'iwslt17/tst2016': {
        'data': [
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/en/fr/en-fr.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/fr/en/fr-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/en/de/en-de.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/de/en/de-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/en/zh/en-zh.tgz',
            'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/zh/en/zh-en.tgz',
        ],
        "md5": [
            "1849bcc3b006dc0642a8843b11aa7192",
            "79bf7a2ef02d226875f55fb076e7e473",
            "b68e7097b179491f6c466ef41ad72b9b",
            "e3f5b2a075a2da1a395c8b60bf1e9be1",
            "975a858783a0ebec8c57d83ddd5bd381",
            "cc51d9b7fe1ff2af858c6a0dd80b8815",
        ],
        'description': 'Development data for IWSLT 2017.',
        'en-fr': ['en-fr/IWSLT17.TED.tst2016.en-fr.en.xml', 'fr-en/IWSLT17.TED.tst2016.fr-en.fr.xml'],
        'fr-en': ['fr-en/IWSLT17.TED.tst2016.fr-en.fr.xml', 'en-fr/IWSLT17.TED.tst2016.en-fr.en.xml'],
        'en-de': ['en-de/IWSLT17.TED.tst2016.en-de.en.xml', 'de-en/IWSLT17.TED.tst2016.de-en.de.xml'],
        'de-en': ['de-en/IWSLT17.TED.tst2016.de-en.de.xml', 'en-de/IWSLT17.TED.tst2016.en-de.en.xml'],
        'en-zh': ['en-zh/IWSLT17.TED.tst2016.en-zh.en.xml', 'zh-en/IWSLT17.TED.tst2016.zh-en.zh.xml'],
        'zh-en': ['zh-en/IWSLT17.TED.tst2016.zh-en.zh.xml', 'en-zh/IWSLT17.TED.tst2016.en-zh.en.xml'],
    },
    'iwslt17/tst2015': {
        'data': [
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/de/en-de.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/fr/en-fr.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/zh/en-zh.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/zh/en/zh-en.tgz',
        ],
        "md5": [
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        'description': 'Development data for IWSLT 2017.',
        'en-fr': ['en-fr/IWSLT17.TED.tst2015.en-fr.en.xml', 'fr-en/IWSLT17.TED.tst2015.fr-en.fr.xml'],
        'fr-en': ['fr-en/IWSLT17.TED.tst2015.fr-en.fr.xml', 'en-fr/IWSLT17.TED.tst2015.en-fr.en.xml'],
        'en-de': ['en-de/IWSLT17.TED.tst2015.en-de.en.xml', 'de-en/IWSLT17.TED.tst2015.de-en.de.xml'],
        'de-en': ['de-en/IWSLT17.TED.tst2015.de-en.de.xml', 'en-de/IWSLT17.TED.tst2015.en-de.en.xml'],
        'en-zh': ['en-zh/IWSLT17.TED.tst2015.en-zh.en.xml', 'zh-en/IWSLT17.TED.tst2015.zh-en.zh.xml'],
        'zh-en': ['zh-en/IWSLT17.TED.tst2015.zh-en.zh.xml', 'en-zh/IWSLT17.TED.tst2015.en-zh.en.xml'],
    },
    'iwslt17/tst2014': {
        'data': [
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/de/en-de.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/fr/en-fr.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/zh/en-zh.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/zh/en/zh-en.tgz',
        ],
        "md5": [
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        'description': 'Development data for IWSLT 2017.',
        'en-fr': ['en-fr/IWSLT17.TED.tst2014.en-fr.en.xml', 'fr-en/IWSLT17.TED.tst2014.fr-en.fr.xml'],
        'fr-en': ['fr-en/IWSLT17.TED.tst2014.fr-en.fr.xml', 'en-fr/IWSLT17.TED.tst2014.en-fr.en.xml'],
        'en-de': ['en-de/IWSLT17.TED.tst2014.en-de.en.xml', 'de-en/IWSLT17.TED.tst2014.de-en.de.xml'],
        'de-en': ['de-en/IWSLT17.TED.tst2014.de-en.de.xml', 'en-de/IWSLT17.TED.tst2014.en-de.en.xml'],
        'en-zh': ['en-zh/IWSLT17.TED.tst2014.en-zh.en.xml', 'zh-en/IWSLT17.TED.tst2014.zh-en.zh.xml'],
        'zh-en': ['zh-en/IWSLT17.TED.tst2014.zh-en.zh.xml', 'en-zh/IWSLT17.TED.tst2014.en-zh.en.xml'],
    },
    'iwslt17/tst2013': {
        'data': [
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/de/en-de.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/fr/en-fr.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/zh/en-zh.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/zh/en/zh-en.tgz',
        ],
        "md5": [
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        'description': 'Development data for IWSLT 2017.',
        'en-fr': ['en-fr/IWSLT17.TED.tst2013.en-fr.en.xml', 'fr-en/IWSLT17.TED.tst2013.fr-en.fr.xml'],
        'fr-en': ['fr-en/IWSLT17.TED.tst2013.fr-en.fr.xml', 'en-fr/IWSLT17.TED.tst2013.en-fr.en.xml'],
        'en-de': ['en-de/IWSLT17.TED.tst2013.en-de.en.xml', 'de-en/IWSLT17.TED.tst2013.de-en.de.xml'],
        'de-en': ['de-en/IWSLT17.TED.tst2013.de-en.de.xml', 'en-de/IWSLT17.TED.tst2013.en-de.en.xml'],
        'en-zh': ['en-zh/IWSLT17.TED.tst2013.en-zh.en.xml', 'zh-en/IWSLT17.TED.tst2013.zh-en.zh.xml'],
        'zh-en': ['zh-en/IWSLT17.TED.tst2013.zh-en.zh.xml', 'en-zh/IWSLT17.TED.tst2013.en-zh.en.xml'],
    },
    'iwslt17/tst2012': {
        'data': [
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/de/en-de.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/fr/en-fr.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/zh/en-zh.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/zh/en/zh-en.tgz',
        ],
        "md5": [
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        'description': 'Development data for IWSLT 2017.',
        'en-fr': ['en-fr/IWSLT17.TED.tst2012.en-fr.en.xml', 'fr-en/IWSLT17.TED.tst2012.fr-en.fr.xml'],
        'fr-en': ['fr-en/IWSLT17.TED.tst2012.fr-en.fr.xml', 'en-fr/IWSLT17.TED.tst2012.en-fr.en.xml'],
        'en-de': ['en-de/IWSLT17.TED.tst2012.en-de.en.xml', 'de-en/IWSLT17.TED.tst2012.de-en.de.xml'],
        'de-en': ['de-en/IWSLT17.TED.tst2012.de-en.de.xml', 'en-de/IWSLT17.TED.tst2012.en-de.en.xml'],
        'en-zh': ['en-zh/IWSLT17.TED.tst2012.en-zh.en.xml', 'zh-en/IWSLT17.TED.tst2012.zh-en.zh.xml'],
        'zh-en': ['zh-en/IWSLT17.TED.tst2012.zh-en.zh.xml', 'en-zh/IWSLT17.TED.tst2012.en-zh.en.xml'],
    },
    'iwslt17/tst2011': {
        'data': [
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/de/en-de.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/fr/en-fr.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/zh/en-zh.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/zh/en/zh-en.tgz',
        ],
        "md5": [
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        'description': 'Development data for IWSLT 2017.',
        'en-fr': ['en-fr/IWSLT17.TED.tst2011.en-fr.en.xml', 'fr-en/IWSLT17.TED.tst2011.fr-en.fr.xml'],
        'fr-en': ['fr-en/IWSLT17.TED.tst2011.fr-en.fr.xml', 'en-fr/IWSLT17.TED.tst2011.en-fr.en.xml'],
        'en-de': ['en-de/IWSLT17.TED.tst2011.en-de.en.xml', 'de-en/IWSLT17.TED.tst2011.de-en.de.xml'],
        'de-en': ['de-en/IWSLT17.TED.tst2011.de-en.de.xml', 'en-de/IWSLT17.TED.tst2011.en-de.en.xml'],
        'en-zh': ['en-zh/IWSLT17.TED.tst2011.en-zh.en.xml', 'zh-en/IWSLT17.TED.tst2011.zh-en.zh.xml'],
        'zh-en': ['zh-en/IWSLT17.TED.tst2011.zh-en.zh.xml', 'en-zh/IWSLT17.TED.tst2011.en-zh.en.xml'],
    },
    'iwslt17/tst2010': {
        'data': [
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/de/en-de.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/fr/en-fr.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/zh/en-zh.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/zh/en/zh-en.tgz',
        ],
        "md5": [
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        'description': 'Development data for IWSLT 2017.',
        'en-fr': ['en-fr/IWSLT17.TED.tst2010.en-fr.en.xml', 'fr-en/IWSLT17.TED.tst2010.fr-en.fr.xml'],
        'fr-en': ['fr-en/IWSLT17.TED.tst2010.fr-en.fr.xml', 'en-fr/IWSLT17.TED.tst2010.en-fr.en.xml'],
        'en-de': ['en-de/IWSLT17.TED.tst2010.en-de.en.xml', 'de-en/IWSLT17.TED.tst2010.de-en.de.xml'],
        'de-en': ['de-en/IWSLT17.TED.tst2010.de-en.de.xml', 'en-de/IWSLT17.TED.tst2010.en-de.en.xml'],
        'en-zh': ['en-zh/IWSLT17.TED.tst2010.en-zh.en.xml', 'zh-en/IWSLT17.TED.tst2010.zh-en.zh.xml'],
        'zh-en': ['zh-en/IWSLT17.TED.tst2010.zh-en.zh.xml', 'en-zh/IWSLT17.TED.tst2010.en-zh.en.xml'],
    },
    'iwslt17/dev2010': {
        'data': [
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/de/en-de.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/fr/en-fr.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/zh/en-zh.tgz',
            'https://wit3.fbk.eu/archive/2017-01-trnted/texts/zh/en/zh-en.tgz',
        ],
        "md5": [
            "d8a32cfc002a4f12b17429cfa78050e6",
            "ca2b94d694150d4d6c5dc64c200fa589",
            "3cf07ebe305312b12f7f1a4d5f8f8377",
            "19927da9de0f40348cad9c0fc61642ac",
            "575b788dad6c5b9c5cee636f9ac1094a",
            "1c0ae40171d52593df8a6963d3828116",
        ],
        'description': 'Development data for IWSLT 2017.',
        'en-fr': ['en-fr/IWSLT17.TED.dev2010.en-fr.en.xml', 'fr-en/IWSLT17.TED.dev2010.fr-en.fr.xml'],
        'fr-en': ['fr-en/IWSLT17.TED.dev2010.fr-en.fr.xml', 'en-fr/IWSLT17.TED.dev2010.en-fr.en.xml'],
        'en-de': ['en-de/IWSLT17.TED.dev2010.en-de.en.xml', 'de-en/IWSLT17.TED.dev2010.de-en.de.xml'],
        'de-en': ['de-en/IWSLT17.TED.dev2010.de-en.de.xml', 'en-de/IWSLT17.TED.dev2010.en-de.en.xml'],
        'en-zh': ['en-zh/IWSLT17.TED.dev2010.en-zh.en.xml', 'zh-en/IWSLT17.TED.dev2010.zh-en.zh.xml'],
        'zh-en': ['zh-en/IWSLT17.TED.dev2010.zh-en.zh.xml', 'en-zh/IWSLT17.TED.dev2010.en-zh.en.xml'],
    },
}


# pep8: enable=E501


def tokenize_13a(line):
    """
    Tokenizes an input line using a relatively minimal tokenization that is
    however equivalent to mteval-v13a, used by WMT.

    :param line: a segment to tokenize
    :return: the tokenized line
    """

    norm = line

    # language-independent part:
    norm = norm.replace('<skipped>', '')
    # norm = norm.replace('-\n', '')
    norm = norm.replace('\n', ' ')
    norm = norm.replace('&quot;', '"')
    norm = norm.replace('&amp;', '&')
    norm = norm.replace('&lt;', '<')
    norm = norm.replace('&gt;', '>')

    # language-dependent part (assuming Western languages):
    norm = " {} ".format(norm)
    norm = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', ' \\1 ', norm)
    norm = re.sub(r'([^0-9])([\.,])', '\\1 \\2 ', norm)  # tokenize period and comma unless preceded by a digit
    norm = re.sub(r'([\.,])([^0-9])', ' \\1 \\2', norm)  # tokenize period and comma unless followed by a digit
    norm = re.sub(r'([0-9])(-)', '\\1 \\2 ', norm)  # tokenize dash when preceded by a digit
    norm = re.sub(r'\s+', ' ', norm)  # one space only between words
    norm = re.sub(r'^\s+', '', norm)  # no leading space
    norm = re.sub(r'\s+$', '', norm)  # no trailing space

    return norm


class UnicodeRegex:
    """Ad-hoc hack to recognize all punctuation and symbols.

    without depending on https://pypi.python.org/pypi/regex/."""

    def _property_chars(prefix):
        return ''.join(chr(x) for x in range(sys.maxunicode) if unicodedata.category(chr(x)).startswith(prefix))

    punctuation = _property_chars('P')
    nondigit_punct_re = re.compile(r'([^\d])([' + punctuation + r'])')
    punct_nondigit_re = re.compile(r'([' + punctuation + r'])([^\d])')
    symbol_re = re.compile('([' + _property_chars('S') + '])')


def tokenize_v14_international(string):
    # pep8: disable=E501
    r"""Tokenize a string following the official BLEU implementation.

    See
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be
    just one line and no HTML entities de-escaping is needed. So we just
    tokenize on punctuation and symbols, except when a punctuation is
    preceded and followed by a digit (e.g. a comma/dot as a thousand/decimal
    separator).

    Note that a number (e.g., a year) followed by a dot at the end of
    sentence is NOT tokenized, i.e. the dot stays with the number because
    `s/(\p{P})(\P{N})/ $1 $2/g` does not match this case (unless we add a
    space after each sentence). However, this error is already in the
    original mteval-v14.pl and we want to be consistent with it. The error is
    not present in the non-international version, which uses `$norm_text = "
    $norm_text "` (or `norm = " {} ".format(norm)` in Python).

    :param string: the input string
    :return: a list of tokens
    """
    # pep8: enable=E501

    string = UnicodeRegex.nondigit_punct_re.sub(r'\1 \2 ', string)
    string = UnicodeRegex.punct_nondigit_re.sub(r' \1 \2', string)
    string = UnicodeRegex.symbol_re.sub(r' \1 ', string)
    return string.strip()


def tokenize_zh(sentence):
    """MIT License
    Copyright (c) 2017 - Shujian Huang <huangsj@nju.edu.cn>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.

    The tokenization of Chinese text in this script contains two steps:
    separate each Chinese characters (by utf-8 encoding); tokenize the non
    Chinese part (following the mteval script). Author: Shujian Huang
    huangsj@nju.edu.cn

    :param sentence: input sentence
    :return: tokenized sentence
    """

    def is_chinese_char(uchar):
        """
        :param uchar: input char in unicode
        :return: whether the input char is a Chinese character.
        """
        # CJK Unified Ideographs Extension A, release 3.0
        if u'\u3400' <= uchar <= u'\u4db5':
            return True
        # CJK Unified Ideographs, release 1.1
        elif u'\u4e00' <= uchar <= u'\u9fa5':
            return True
        # CJK Unified Ideographs, release 4.1
        elif u'\u9fa6' <= uchar <= u'\u9fbb':
            return True
        # CJK Compatibility Ideographs, release 1.1
        elif u'\uf900' <= uchar <= u'\ufa2d':
            return True
        # CJK Compatibility Ideographs, release 3.2
        elif u'\ufa30' <= uchar <= u'\ufa6a':
            return True
        # CJK Compatibility Ideographs, release 4.1
        elif u'\ufa70' <= uchar <= u'\ufad9':
            return True
        # CJK Unified Ideographs Extension B, release 3.1
        elif u'\u20000' <= uchar <= u'\u2a6d6':
            return True
        # CJK Compatibility Supplement, release 3.1
        elif u'\u2f800' <= uchar <= u'\u2fa1d':
            return True
        # Full width ASCII, full width of English punctuation, half width
        # Katakana, half wide half width kana, Korean alphabet
        elif u'\uff00' <= uchar <= u'\uffef':
            return True
        # CJK Radicals Supplement
        elif u'\u2e80' <= uchar <= u'\u2eff':
            return True
        # CJK punctuation mark
        elif u'\u3000' <= uchar <= u'\u303f':
            return True
        # CJK stroke
        elif u'\u31c0' <= uchar <= u'\u31ef':
            return True
        # Kangxi Radicals
        elif u'\u2f00' <= uchar <= u'\u2fdf':
            return True
        # Chinese character structure
        elif u'\u2ff0' <= uchar <= u'\u2fff':
            return True
        # Phonetic symbols
        elif u'\u3100' <= uchar <= u'\u312f':
            return True
        # Phonetic symbols (Taiwanese and Hakka expansion)
        elif u'\u31a0' <= uchar <= u'\u31bf':
            return True
        elif u'\ufe10' <= uchar <= u'\ufe1f':
            return True
        elif u'\ufe30' <= uchar <= u'\ufe4f':
            return True
        elif u'\u2600' <= uchar <= u'\u26ff':
            return True
        elif u'\u2700' <= uchar <= u'\u27bf':
            return True
        elif u'\u3200' <= uchar <= u'\u32ff':
            return True
        elif u'\u3300' <= uchar <= u'\u33ff':
            return True

        return False

    sentence = sentence.strip()
    sentence_in_chars = ""
    for char in sentence:
        if is_chinese_char(char):
            sentence_in_chars += " "
            sentence_in_chars += char
            sentence_in_chars += " "
        else:
            sentence_in_chars += char
    sentence = sentence_in_chars

    # TODO: the code above could probably be replaced with the following line:
    # import regex
    # sentence = regex.sub(r'(\p{Han})', r' \1 ', sentence)

    # tokenize punctuation
    sentence = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 ', sentence)

    # tokenize period and comma unless preceded by a digit
    sentence = re.sub(r'([^0-9])([\.,])', r'\1 \2 ', sentence)

    # tokenize period and comma unless followed by a digit
    sentence = re.sub(r'([\.,])([^0-9])', r' \1 \2', sentence)

    # tokenize dash when preceded by a digit
    sentence = re.sub(r'([0-9])(-)', r'\1 \2 ', sentence)

    # one space only between words
    sentence = re.sub(r'\s+', r' ', sentence)

    # no leading or trailing spaces
    sentence = sentence.strip()

    return sentence


TOKENIZERS = {
    '13a': tokenize_13a,
    'intl': tokenize_v14_international,
    'zh': tokenize_zh,
    'fairseq': tokenize_en,
    'none': lambda x: x,
}
DEFAULT_TOKENIZER = '13a'


def smart_open(file, mode='rt', encoding='utf-8'):
    """Convenience function for reading compressed or plain text files.
    :param file: The file to read.
    :param encoding: The file encoding.
    """
    if file.endswith('.gz'):
        return gzip.open(file, mode=mode, encoding=encoding, newline="\n")
    return open(file, mode=mode, encoding=encoding, newline="\n")


def my_log(num):
    """
    Floors the log function

    :param num: the number
    :return: log(num) floored to a very low number
    """

    if num == 0.0:
        return -9999999999
    return math.log(num)


def bleu_signature(args, numrefs):
    """
    Builds a signature that uniquely identifies the scoring parameters used.
    :param args: the arguments passed into the script
    :return: the signature
    """

    # Abbreviations for the signature
    abbr = {'test': 't', 'lang': 'l', 'smooth': 's', 'case': 'c', 'tok': 'tok', 'numrefs': '#', 'version': 'v'}

    signature = {
        'tok': args.tokenize,
        'version': VERSION,
        'smooth': args.smooth,
        'numrefs': numrefs,
        'case': 'lc' if args.lc else 'mixed',
    }

    if args.test_set is not None:
        signature['test'] = args.test_set

    if args.langpair is not None:
        signature['lang'] = args.langpair

    sigstr = '+'.join(['{}.{}'.format(abbr[x] if args.short else x, signature[x]) for x in sorted(signature.keys())])

    return sigstr


def chrf_signature(args, numrefs):
    """
    Builds a signature that uniquely identifies the scoring parameters used.
    :param args: the arguments passed into the script
    :return: the chrF signature
    """

    # Abbreviations for the signature
    abbr = {'test': 't', 'lang': 'l', 'numchars': 'n', 'space': 's', 'case': 'c', 'numrefs': '#', 'version': 'v'}

    signature = {
        'tok': args.tokenize,
        'version': VERSION,
        'space': args.chrf_whitespace,
        'numchars': args.chrf_order,
        'numrefs': numrefs,
        'case': 'lc' if args.lc else 'mixed',
    }

    if args.test_set is not None:
        signature['test'] = args.test_set

    if args.langpair is not None:
        signature['lang'] = args.langpair

    sigstr = '+'.join(['{}.{}'.format(abbr[x] if args.short else x, signature[x]) for x in sorted(signature.keys())])

    return sigstr


def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> Counter:
    """Extracts all the ngrams (1 <= n <= NGRAM_ORDER) from a sequence of tokens

    :param line: a segment containing a sequence of words
    :param max_order: collect n-grams from 1<=n<=max
    :return: a dictionary containing ngrams and counts
    """

    ngrams = Counter()
    tokens = line.split()
    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            ngram = ' '.join(tokens[i : i + n])
            ngrams[ngram] += 1

    return ngrams


def extract_char_ngrams(s: str, n: int) -> Counter:
    """
    Yields counts of character n-grams from string s of order n.
    """
    return Counter([s[i : i + n] for i in range(len(s) - n + 1)])


def ref_stats(output, refs):
    ngrams = Counter()
    closest_diff = None
    closest_len = None
    for ref in refs:
        tokens = ref.split()
        reflen = len(tokens)
        diff = abs(len(output.split()) - reflen)
        if closest_diff is None or diff < closest_diff:
            closest_diff = diff
            closest_len = reflen
        elif diff == closest_diff:
            if reflen < closest_len:
                closest_len = reflen

        ngrams_ref = extract_ngrams(ref)
        for ngram in ngrams_ref.keys():
            ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

    return ngrams, closest_diff, closest_len


def _clean(s):
    """
    Removes trailing and leading spaces and collapses multiple consecutive
    internal spaces to a single one.

    :param s: The string.
    :return: A cleaned-up string.
    """
    return re.sub(r'\s+', ' ', s.strip())


def process_to_text(rawfile, txtfile, field: int = None):
    """Processes raw files to plain text files.
    :param rawfile: the input file (possibly SGML)
    :param txtfile: the plaintext file
    :param field: For TSV files, which field to extract.
    """

    if not os.path.exists(txtfile) or os.path.getsize(txtfile) == 0:
        logging.info("Processing %s to %s", rawfile, txtfile)
        if rawfile.endswith('.sgm') or rawfile.endswith('.sgml'):
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    if line.startswith('<seg '):
                        logging.info(_clean(re.sub(r'<seg.*?>(.*)</seg>.*?', '\\1', line)), file=fout)
        elif rawfile.endswith('.xml'):  # IWSLT
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    if line.startswith('<seg '):
                        logging.info(_clean(re.sub(r'<seg.*?>(.*)</seg>.*?', '\\1', line)), file=fout)
        elif rawfile.endswith('.txt'):  # wmt17/ms
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    logging.info(line.rstrip(), file=fout)
        elif rawfile.endswith('.tsv'):  # MTNT
            with smart_open(rawfile) as fin, smart_open(txtfile, 'wt') as fout:
                for line in fin:
                    logging.info(line.rstrip().split('\t')[field], file=fout)


def print_test_set(test_set, langpair, side):
    """Prints to STDOUT the specified side of the specified test set
    :param test_set: the test set to print
    :param langpair: the language pair
    :param side: 'src' for source, 'ref' for reference
    """

    files = download_test_set(test_set, langpair)
    if side == 'src':
        files = [files[0]]
    elif side == 'ref':
        files.pop(0)

    streams = [smart_open(file) for file in files]
    for lines in zip(*streams):
        logging.info('\t'.join(map(lambda x: x.rstrip(), lines)))


def download_test_set(test_set, langpair=None):
    """Downloads the specified test to the system location specified by the
    SACREBLEU environment variable. :param test_set: the test set to download
    :param langpair: the language pair (needed for some datasets) :return:
    the set of processed files
    """

    outdir = os.path.join(SACREBLEU_DIR, test_set)
    if not os.path.exists(outdir):
        logging.info('Creating %s', outdir)
        os.makedirs(outdir)

    expected_checksums = DATASETS[test_set].get('md5', [None] * len(DATASETS[test_set]))
    for dataset, expected_md5 in zip(DATASETS[test_set]['data'], expected_checksums):
        tarball = os.path.join(outdir, os.path.basename(dataset))
        rawdir = os.path.join(outdir, 'raw')
        if not os.path.exists(tarball) or os.path.getsize(tarball) == 0:
            logging.info("Downloading %s to %s", dataset, tarball)
            try:
                with urllib.request.urlopen(dataset) as f, open(tarball, 'wb') as out:
                    out.write(f.read())
            except ssl.SSLError:
                logging.warning(
                    'An SSL error was encountered in downloading the files. '
                    'If you\'re on a Mac, '
                    'you may need to run the "Install Certificates.command" '
                    'file located in the '
                    '"Python 3" folder, often found under /Applications'
                )
                sys.exit(1)

            # Check md5sum
            if expected_md5 is not None:
                md5 = hashlib.md5()
                with open(tarball, 'rb') as infile:
                    for line in infile:
                        md5.update(line)
                if md5.hexdigest() != expected_md5:
                    logging.error(
                        'Fatal: MD5 sum of downloaded file was incorrect (got '
                        '{}, expected {}).'.format(md5.hexdigest(), expected_md5)
                    )
                    logging.error('Please manually delete "{}" and rerun the command.'.format(tarball))
                    logging.error(
                        'If the problem persists, the tarball may have '
                        'changed, in which case, please contact the SacreBLEU '
                        'maintainer. '
                    )
                    sys.exit(1)
                else:
                    logging.info('Checksum passed: {}'.format(md5.hexdigest()))

            # Extract the tarball
            logging.info('Extracting %s', tarball)
            if tarball.endswith('.tar.gz') or tarball.endswith('.tgz'):
                import tarfile

                tar = tarfile.open(tarball)
                tar.extractall(path=rawdir)
            elif tarball.endswith('.zip'):
                import zipfile

                zipfile = zipfile.ZipFile(tarball, 'r')
                zipfile.extractall(path=rawdir)
                zipfile.close()

    found = []

    # Process the files into plain text
    languages = DATASETS[test_set].keys() if langpair is None else [langpair]
    for pair in languages:
        if '-' not in pair:
            continue
        src, tgt = pair.split('-')
        rawfile = DATASETS[test_set][pair][0]
        field = None  # used for TSV files
        if rawfile.endswith('.tsv'):
            field, rawfile = rawfile.split(':', maxsplit=1)
            field = int(field)
        rawpath = os.path.join(rawdir, rawfile)
        outpath = os.path.join(outdir, '{}.{}'.format(pair, src))
        process_to_text(rawpath, outpath, field=field)
        found.append(outpath)

        refs = DATASETS[test_set][pair][1:]
        for i, ref in enumerate(refs):
            field = None
            if ref.endswith('.tsv'):
                field, ref = ref.split(':', maxsplit=1)
                field = int(field)
            rawpath = os.path.join(rawdir, ref)
            if len(refs) >= 2:
                outpath = os.path.join(outdir, '{}.{}.{}'.format(pair, tgt, i))
            else:
                outpath = os.path.join(outdir, '{}.{}'.format(pair, tgt))
            process_to_text(rawpath, outpath, field=field)
            found.append(outpath)

    return found


class BLEU(namedtuple('BaseBLEU', 'score, counts, totals, precisions, bp, sys_len, ref_len')):
    def format(self, width=2):
        precisions = "/".join(["{:.1f}".format(p) for p in self.precisions])
        return (
            f'BLEU = {self.score:.{width}f} {precisions}'
            f'(BP = {self.bp:.3f}'
            f' ratio = {(self.sys_len / self.ref_len):.3f}'
            f' hyp_len = {self.sys_len:d}'
            f' ref_len = {self.ref_len:d})'
        )

    def __str__(self):
        return self.format()


def compute_bleu(
    correct: List[int],
    total: List[int],
    sys_len: int,
    ref_len: int,
    smooth_method='none',
    smooth_value=SMOOTH_VALUE_DEFAULT,
    use_effective_order=False,
) -> BLEU:
    """Computes BLEU score from its sufficient statistics. Adds smoothing.

    Smoothing methods (citing "A Systematic Comparison of Smoothing
    Techniques for Sentence-Level BLEU", Boxing Chen and Colin Cherry,
    WMT 2014: http://aclweb.org/anthology/W14-3346)

    - exp: NIST smoothing method (Method 3)
    - floor: Method 1
    - add-k: Method 2 (generalizing Lin and Och, 2004)
    - none: do nothing.

    :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
    :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
    :param sys_len: The cumulative system length :param ref_len: The
    cumulative reference length :param smooth: The smoothing method to use
    :param smooth_value: The smoothing value added, if smooth method 'floor'
    is used :param use_effective_order: Use effective order. :return: A BLEU
    object with the score (100-based) and other statistics.
    """

    precisions = [0 for x in range(NGRAM_ORDER)]

    smooth_mteval = 1.0
    effective_order = NGRAM_ORDER
    for n in range(NGRAM_ORDER):
        if smooth_method == 'add-k' and n > 1:
            correct[n] += smooth_value
            total[n] += smooth_value
        if total[n] == 0:
            break

        if use_effective_order:
            effective_order = n + 1

        if correct[n] == 0:
            if smooth_method == 'exp':
                smooth_mteval *= 2
                precisions[n] = 100.0 / (smooth_mteval * total[n])
            elif smooth_method == 'floor':
                precisions[n] = 100.0 * smooth_value / total[n]
        else:
            precisions[n] = 100.0 * correct[n] / total[n]

    # If the system guesses no i-grams, 1 <= i <= NGRAM_ORDER, the BLEU score
    # is 0 (technically undefined). This is a problem for sentence-level BLEU
    # or a corpus of short sentences, where systems will get no credit if
    # sentence lengths fall under the NGRAM_ORDER threshold. This fix scales
    # NGRAM_ORDER to the observed maximum order. It is only available through
    # the API and off by default

    brevity_penalty = 1.0
    if sys_len < ref_len:
        brevity_penalty = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0

    bleu = brevity_penalty * math.exp(sum(map(my_log, precisions[:effective_order])) / effective_order)

    return BLEU._make([bleu, correct, total, precisions, brevity_penalty, sys_len, ref_len])


def sentence_bleu(
    hypothesis: str,
    reference: str,
    smooth_method: str = 'floor',
    smooth_value: float = SMOOTH_VALUE_DEFAULT,
    use_effective_order: bool = True,
):
    """
    Computes BLEU on a single sentence pair.

    Disclaimer: computing BLEU on the sentence level is not its intended use,
    BLEU is a corpus-level metric.

    :param hypothesis: Hypothesis string. :param reference: Reference string.
    :param smooth_value: For 'floor' smoothing, the floor value to use.
    :param use_effective_order: Account for references that are shorter than
    the largest n-gram. :return: Returns a single BLEU score as a float.
    """
    bleu = corpus_bleu(
        hypothesis,
        reference,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        use_effective_order=use_effective_order,
    )
    return bleu.score


def corpus_bleu(
    sys_stream: Union[str, Iterable[str]],
    ref_streams: Union[str, List[Iterable[str]]],
    smooth_method='exp',
    smooth_value=SMOOTH_VALUE_DEFAULT,
    force=False,
    lowercase=False,
    tokenize=DEFAULT_TOKENIZER,
    use_effective_order=False,
) -> BLEU:
    """Produces BLEU scores along with its sufficient statistics from a
    source against one or more references.

    :param sys_stream: The system stream (a sequence of segments) :param
    ref_streams: A list of one or more reference streams (each a sequence of
    segments) :param smooth: The smoothing method to use :param smooth_value:
    For 'floor' smoothing, the floor to use :param force: Ignore data that
    looks already tokenized :param lowercase: Lowercase the data :param
    tokenize: The tokenizer to use :return: a BLEU object containing
    everything you'd want
    """

    # Add some robustness to the input arguments
    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    sys_len = 0
    ref_len = 0

    correct = [0 for n in range(NGRAM_ORDER)]
    total = [0 for n in range(NGRAM_ORDER)]

    # look for already-tokenized sentences
    tokenized_count = 0

    fhs = [sys_stream] + ref_streams
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")

        if lowercase:
            lines = [x.lower() for x in lines]

        if not (force or tokenize == 'none') and lines[0].rstrip().endswith(' .'):
            tokenized_count += 1

            if tokenized_count == 100:
                logging.warning('That\'s 100 lines that end in a tokenized period (\'.\')')
                logging.warning(
                    'It looks like you forgot to detokenize your test data, ' 'which may hurt your score. '
                )
                logging.warning(
                    'If you insist your data is detokenized, or don\'t care, '
                    'you can suppress this message with \'--force\'. '
                )

        output, *refs = [TOKENIZERS[tokenize](x.rstrip()) for x in lines]

        ref_ngrams, closest_diff, closest_len = ref_stats(output, refs)

        sys_len += len(output.split())
        ref_len += closest_len

        sys_ngrams = extract_ngrams(output)
        for ngram in sys_ngrams.keys():
            n = len(ngram.split())
            correct[n - 1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
            total[n - 1] += sys_ngrams[ngram]

    return compute_bleu(
        correct,
        total,
        sys_len,
        ref_len,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        use_effective_order=use_effective_order,
    )


def raw_corpus_bleu(sys_stream, ref_streams, smooth_value=SMOOTH_VALUE_DEFAULT) -> BLEU:
    """Convenience function that wraps corpus_bleu(). This is convenient if
    you're using sacrebleu as a library, say for scoring on dev. It uses no
    tokenization and 'floor' smoothing, with the floor default to 0 (no
    smoothing).

    :param sys_stream: the system stream (a sequence of segments) :param
    ref_streams: a list of one or more reference streams (each a sequence of
    segments)
    """
    return corpus_bleu(
        sys_stream,
        ref_streams,
        smooth_method='floor',
        smooth_value=smooth_value,
        force=True,
        tokenize='none',
        use_effective_order=True,
    )


def delete_whitespace(text: str) -> str:
    """
    Removes whitespaces from text.
    """
    return re.sub(r'\s+', '', text).strip()


def get_sentence_statistics(
    hypothesis: str, reference: str, order: int = CHRF_ORDER, remove_whitespace: bool = True
) -> List[float]:
    hypothesis = delete_whitespace(hypothesis) if remove_whitespace else hypothesis
    reference = delete_whitespace(reference) if remove_whitespace else reference
    statistics = [0] * (order * 3)
    for i in range(order):
        n = i + 1
        hypothesis_ngrams = extract_char_ngrams(hypothesis, n)
        reference_ngrams = extract_char_ngrams(reference, n)
        common_ngrams = hypothesis_ngrams & reference_ngrams
        statistics[3 * i + 0] = sum(hypothesis_ngrams.values())
        statistics[3 * i + 1] = sum(reference_ngrams.values())
        statistics[3 * i + 2] = sum(common_ngrams.values())
    return statistics


def get_corpus_statistics(
    hypotheses: Iterable[str], references: Iterable[str], order: int = CHRF_ORDER, remove_whitespace: bool = True
) -> List[float]:
    corpus_statistics = [0] * (order * 3)
    for hypothesis, reference in zip(hypotheses, references):
        statistics = get_sentence_statistics(hypothesis, reference, order=order, remove_whitespace=remove_whitespace)
        for i in range(len(statistics)):
            corpus_statistics[i] += statistics[i]
    return corpus_statistics


def _avg_precision_and_recall(statistics: List[float], order: int) -> Tuple[float, float]:
    avg_precision = 0.0
    avg_recall = 0.0
    effective_order = 0
    for i in range(order):
        hypotheses_ngrams = statistics[3 * i + 0]
        references_ngrams = statistics[3 * i + 1]
        common_ngrams = statistics[3 * i + 2]
        if hypotheses_ngrams > 0 and references_ngrams > 0:
            avg_precision += common_ngrams / hypotheses_ngrams
            avg_recall += common_ngrams / references_ngrams
            effective_order += 1
    if effective_order == 0:
        return 0.0, 0.0
    avg_precision /= effective_order
    avg_recall /= effective_order
    return avg_precision, avg_recall


def _chrf(avg_precision, avg_recall, beta: int = CHRF_BETA) -> float:
    if avg_precision + avg_recall == 0:
        return 0.0
    beta_square = beta ** 2
    score = (1 + beta_square) * (avg_precision * avg_recall) / ((beta_square * avg_precision) + avg_recall)
    return score


def corpus_chrf(
    hypotheses: Iterable[str],
    references: Iterable[str],
    order: int = CHRF_ORDER,
    beta: float = CHRF_BETA,
    remove_whitespace: bool = True,
) -> float:
    """
    Computes Chrf on a corpus.

    :param hypotheses: Stream of hypotheses. :param references: Stream of
    references :param order: Maximum n-gram order. :param remove_whitespace:
    Whether to delete all whitespace from hypothesis and reference strings.
    :param beta: Defines importance of recall w.r.t precision. If beta=1,
    same importance. :return: Chrf score.
    """
    corpus_statistics = get_corpus_statistics(hypotheses, references, order=order, remove_whitespace=remove_whitespace)
    avg_precision, avg_recall = _avg_precision_and_recall(corpus_statistics, order)
    return _chrf(avg_precision, avg_recall, beta=beta)


def sentence_chrf(
    hypothesis: str, reference: str, order: int = CHRF_ORDER, beta: float = CHRF_BETA, remove_whitespace: bool = True
) -> float:
    """
    Computes ChrF on a single sentence pair.

    :param hypothesis: Hypothesis string. :param reference: Reference string.
    :param order: Maximum n-gram order. :param remove_whitespace: Whether to
    delete whitespaces from hypothesis and reference strings. :param beta:
    Defines importance of recall w.r.t precision. If beta=1, same importance.
    :return: Chrf score.
    """
    statistics = get_sentence_statistics(hypothesis, reference, order=order, remove_whitespace=remove_whitespace)
    avg_precision, avg_recall = _avg_precision_and_recall(statistics, order)
    return _chrf(avg_precision, avg_recall, beta=beta)


def main():
    arg_parser = argparse.ArgumentParser(
        description='sacreBLEU: Hassle-free computation of shareable BLEU '
        'scores. Quick usage: score your detokenized output '
        'against WMT\'14 EN-DE: '
        '    cat output.detok.de | ./sacreBLEU -t wmt14 -l en-de'
    )
    arg_parser.add_argument(
        '--test-set', '-t', type=str, default=None, choices=DATASETS.keys(), help='the test set to use'
    )
    arg_parser.add_argument(
        '-lc', action='store_true', default=False, help='use case-insensitive BLEU (default: actual case)'
    )
    arg_parser.add_argument(
        '--smooth',
        '-s',
        choices=['exp', 'floor', 'add-n', 'none'],
        default='exp',
        help='smoothing method: exponential decay (default), floor (increment '
        'zero counts), add-k (increment num/denom by k for n>1), or none ',
    )
    arg_parser.add_argument(
        '--smooth-value',
        '-sv',
        type=float,
        default=SMOOTH_VALUE_DEFAULT,
        help='The value to pass to the smoothing technique, when relevant. ' 'Default: %(default)s. ',
    )
    arg_parser.add_argument(
        '--tokenize', '-tok', choices=TOKENIZERS.keys(), default=None, help='tokenization method to use'
    )
    arg_parser.add_argument(
        '--language-pair',
        '-l',
        dest='langpair',
        default=None,
        help='source-target language pair (2-char ISO639-1 codes)',
    )
    arg_parser.add_argument('--download', type=str, default=None, help='download a test set and quit')
    arg_parser.add_argument(
        '--echo',
        choices=['src', 'ref', 'both'],
        type=str,
        default=None,
        help='output the source (src), reference (ref), or both (both, ' 'pasted) to STDOUT and quit ',
    )
    arg_parser.add_argument('--input', '-i', type=str, default='-', help='Read input from a file instead of STDIN')
    arg_parser.add_argument(
        'refs',
        nargs='*',
        default=[],
        help='optional list of references (for backwards-compatibility with ' 'older scripts) ',
    )
    arg_parser.add_argument(
        '--metrics',
        '-m',
        choices=['bleu', 'chrf'],
        nargs='+',
        default=['bleu'],
        help='metrics to compute (default: bleu)',
    )
    arg_parser.add_argument(
        '--chrf-order', type=int, default=CHRF_ORDER, help='chrf character order (default: %(default)s)'
    )
    arg_parser.add_argument(
        '--chrf-beta', type=int, default=CHRF_BETA, help='chrf BETA parameter (default: %(default)s)'
    )
    arg_parser.add_argument(
        '--chrf-whitespace',
        action='store_true',
        default=False,
        help='include whitespace in chrF calculation (default: %(default)s)',
    )
    arg_parser.add_argument(
        '--short', default=False, action='store_true', help='produce a shorter (less human readable) signature'
    )
    arg_parser.add_argument(
        '--score-only', '-b', default=False, action='store_true', help='output only the BLEU score'
    )
    arg_parser.add_argument(
        '--force', default=False, action='store_true', help='insist that your tokenized input is actually detokenized'
    )
    arg_parser.add_argument('--quiet', '-q', default=False, action='store_true', help='suppress informative output')
    arg_parser.add_argument(
        '--encoding',
        '-e',
        type=str,
        default='utf-8',
        help='open text files with specified encoding (default: %(default)s)',
    )
    arg_parser.add_argument(
        '--citation', '--cite', default=False, action='store_true', help='dump the bibtex citation and quit.'
    )
    arg_parser.add_argument('--width', '-w', type=int, default=1, help='floating point width (default: %(default)s)')
    arg_parser.add_argument('-V', '--version', action='version', version='%(prog)s {}'.format(VERSION))
    args = arg_parser.parse_args()

    # Explicitly set the encoding
    sys.stdin = open(sys.stdin.fileno(), mode='r', encoding='utf-8', buffering=True, newline="\n")
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=True)

    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format='sacreBLEU: %(message)s')

    if args.download:
        download_test_set(args.download, args.langpair)
        sys.exit(0)

    if args.citation:
        if not args.test_set:
            logging.error('I need a test set (-t).')
            sys.exit(1)
        elif 'citation' not in DATASETS[args.test_set]:
            logging.error('No citation found for %s', args.test_set)
            sys.exit(1)

        logging.info(DATASETS[args.test_set]['citation'])
        sys.exit(0)

    if args.test_set is not None and args.test_set not in DATASETS:
        logging.error('The available test sets are: ')
        for testset in sorted(DATASETS.keys(), reverse=True):
            logging.error('  %s: %s', testset, DATASETS[testset].get('description', ''))
        sys.exit(1)

    if args.test_set and (args.langpair is None or args.langpair not in DATASETS[args.test_set]):
        if args.langpair is None:
            logging.error('I need a language pair (-l).')
        elif args.langpair not in DATASETS[args.test_set]:
            logging.error('No such language pair "%s"', args.langpair)
        logging.error(
            'Available language pairs for test set "%s": %s',
            args.test_set,
            ', '.join(filter(lambda x: '-' in x, DATASETS[args.test_set].keys())),
        )
        sys.exit(1)

    if args.echo:
        if args.langpair is None or args.test_set is None:
            logging.warning("--echo requires a test set (--t) and a language pair (-l)")
            sys.exit(1)
        print_test_set(args.test_set, args.langpair, args.echo)
        sys.exit(0)

    if args.test_set is None and len(args.refs) == 0:
        logging.error('I need either a predefined test set (-t) or a list of references')
        logging.error('The available test sets are: ')
        for testset in sorted(DATASETS.keys(), reverse=True):
            logging.error('  %s: %s', testset, DATASETS[testset].get('description', ''))
        sys.exit(1)
    elif args.test_set is not None and len(args.refs) > 0:
        logging.error('I need exactly one of (a) a predefined test set (-t) or (b) a ' 'list of references ')
        sys.exit(1)

    if args.test_set is not None and args.tokenize == 'none':
        logging.warning(
            "You are turning off sacrebleu's internal tokenization ("
            "'--tokenize none'), presumably to supply\n "
            "your own reference tokenization. Published numbers will not be "
            "comparable with other papers.\n"
        )

    # Internal tokenizer settings. Set to 'zh' for Chinese  DEFAULT_TOKENIZER (
    if args.tokenize is None:
        # set default
        if args.langpair is not None and args.langpair.split('-')[1] == 'zh':
            args.tokenize = 'zh'
        else:
            args.tokenize = DEFAULT_TOKENIZER

    if (
        args.langpair is not None
        and args.langpair.split('-')[1] == 'zh'
        and 'bleu' in args.metrics
        and args.tokenize != 'zh'
    ):
        logging.warning('You should also pass "--tok zh" when scoring Chinese...')

    if args.test_set:
        _, *refs = download_test_set(args.test_set, args.langpair)
        if len(refs) == 0:
            logging.info('No references found for test set {}/{}.'.format(args.test_set, args.langpair))
            sys.exit(1)
    else:
        refs = args.refs

    inputfh = (
        io.TextIOWrapper(sys.stdin.buffer, encoding=args.encoding)
        if args.input == '-'
        else smart_open(args.input, encoding=args.encoding)
    )
    system = inputfh.readlines()

    # Read references
    refs = [smart_open(x, encoding=args.encoding).readlines() for x in refs]

    try:
        if 'bleu' in args.metrics:
            bleu = corpus_bleu(
                system,
                refs,
                smooth_method=args.smooth,
                smooth_value=args.smooth_value,
                force=args.force,
                lowercase=args.lc,
                tokenize=args.tokenize,
            )
        if 'chrf' in args.metrics:
            chrf = corpus_chrf(
                system, refs[0], beta=args.chrf_beta, order=args.chrf_order, remove_whitespace=not args.chrf_whitespace
            )
    except EOFError:
        logging.error('The input and reference stream(s) were of different lengths.\n')
        if args.test_set is not None:
            logging.error(
                'This could be a problem with your system output or with '
                'sacreBLEU\'s reference database.\n '
                'If the latter, you can clean out the references cache by '
                'typing:\n '
                '\n'
                '    rm -r %s/%s\n'
                '\n'
                'They will be downloaded automatically again the next time '
                'you run sacreBLEU.',
                SACREBLEU_DIR,
                args.test_set,
            )
        sys.exit(1)

    width = args.width
    for metric in args.metrics:
        if metric == 'bleu':
            if args.score_only:
                logging.info('{0:.{1}f}'.format(bleu.score, width))
            else:
                version_str = bleu_signature(args, len(refs))
                logging.info(bleu.format(width).replace('BLEU', 'BLEU+' + version_str))

        elif metric == 'chrf':
            if args.score_only:
                logging.info('{0:.{1}f}'.format(chrf, width))
            else:
                version_str = chrf_signature(args, len(refs))
                logging.info('chrF{0:d}+{1} = {2:.{3}f}'.format(args.chrf_beta, version_str, chrf, width))


if __name__ == '__main__':
    main()
