""" this code is based on PyTorch's tutorials:
https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
"""
import itertools
import re
import unicodedata

import torch as t

from nemo import logging

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        logging.info(
            "keep_words {} / {} = {:.4f}".format(
                len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index),
            )
        )

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


MAX_LENGTH = 10  # Maximum sentence length to consider


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/response pairs and return a voc object


def readVocs(datafile, corpus_name):
    logging.info("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding="utf-8").read().strip().split("\n")
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH
# threshold


def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(" ")) < MAX_LENGTH and len(p[1].split(" ")) < MAX_LENGTH


# Filter pairs using filterPair condition


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# Using the functions defined above, return a populated voc object and pairs
# list


def loadPrepareData(corpus_name, datafile):
    logging.info("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    logging.info("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    logging.info("Trimmed to {!s} sentence pairs".format(len(pairs)))
    logging.info("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    logging.info("Counted words:", voc.num_words)
    return voc, pairs


MIN_COUNT = 3  # Minimum word count threshold for trimming


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(" "):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(" "):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input
        # or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    # logging.info("Trimmed from {} pairs to {}, {:.4f} of total".format(len(
    # pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths


def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = t.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = t.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length


def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = t.ByteTensor(mask).to(t.bool)
    padVar = t.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs


def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    # return inp.squeeze(-1), output.squeeze(-1)
    return inp, lengths, output, mask, max_target_len
