import os
import re
import string
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import nemo


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def mask_padded_tokens(tokens, pad_id):
    mask = tokens != pad_id
    return mask


def read_intent_slot_outputs(
    queries, intent_file, slot_file, intent_logits, slot_logits, slot_masks, intents=None, slots=None
):
    intent_dict = get_vocab(intent_file)
    slot_dict = get_vocab(slot_file)
    pred_intents = np.argmax(intent_logits, 1)
    pred_slots = np.argmax(slot_logits, axis=2)
    slot_masks = slot_masks > 0.5
    for i, query in enumerate(queries):
        nemo.logging.info(f'Query: {query}')
        pred = pred_intents[i]
        nemo.logging.info(f'Predicted intent:\t{pred}\t{intent_dict[pred]}')
        if intents is not None:
            nemo.logging.info(f'True intent:\t{intents[i]}\t{intent_dict[intents[i]]}')

        pred_slot = pred_slots[i][slot_masks[i]]
        tokens = query.strip().split()

        if len(pred_slot) != len(tokens):
            raise ValueError('Pred_slot and tokens must be of the same length')

        for j, token in enumerate(tokens):
            output = f'{token}\t{slot_dict[pred_slot[j]]}'
            if slots is not None:
                output = f'{output}\t{slot_dict[slots[i][j]]}'
            nemo.logging.info(output)


def get_vocab(file):
    lines = open(file, 'r').readlines()
    lines = [line.strip() for line in lines if line.strip()]
    labels = {i: lines[i] for i in range(len(lines))}
    return labels


def write_vocab(items, outfile):
    vocab = {}
    idx = 0
    with open(outfile, 'w') as f:
        for item in items:
            f.write(item + '\n')
            vocab[item] = idx
            idx += 1
    return vocab


def label2idx(file):
    lines = open(file, 'r').readlines()
    lines = [line.strip() for line in lines if line.strip()]
    labels = {lines[i]: i for i in range(len(lines))}
    return labels


def write_vocab_in_order(vocab, outfile):
    with open(outfile, 'w') as f:
        for key in sorted(vocab.keys()):
            f.write(f'{vocab[key]}\n')


def plot_confusion_matrix(label_ids, labels, preds, graph_fold, normalize=False, prefix=''):
    '''
    Plot confusion matrix.
    Args:
      label_ids (dict): label to id map, for example: {'O': 0, 'LOC': 1}
      labels (list of ints): list of true labels
      preds (list of ints): list of predicted labels
      graph_fold (str): path to output folder
      normalize (bool): flag to indicate whether to normalize confusion matrix
      prefix (str): prefix for the plot name

    '''
    # remove labels from label_ids that don't appear in the dev set
    used_labels = set(labels) | set(preds)
    label_ids = {k: label_ids[k] for k, v in label_ids.items() if v in used_labels}

    ids_to_labels = {label_ids[k]: k for k in label_ids}
    classes = [ids_to_labels[id] for id in sorted(label_ids.values())]

    title = 'Confusion matrix'
    cm = confusion_matrix(labels, preds)
    if normalize:
        sums = cm.sum(axis=1)[:, np.newaxis]
        sums = np.where(sums == 0, 1, sums)
        cm = cm.astype('float') / sums
        title = 'Normalized ' + title

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.matshow(cm)
    ax.set_xticks(np.arange(-1, len(classes) + 1))
    ax.set_yticks(np.arange(-1, len(classes) + 1))
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

    os.makedirs(graph_fold, exist_ok=True)
    fig.colorbar(cax)

    title = (prefix + ' ' + title).strip()
    plt.savefig(os.path.join(graph_fold, title + '_' + time.strftime('%Y%m%d-%H%M%S')))


def list2str(l):
    return ' '.join([str(x) for x in l])


def tensor2list(tensor):
    return tensor.detach().cpu().tolist()


def if_exist(outfold, files):
    if not os.path.exists(outfold):
        return False
    for file in files:
        if not os.path.exists(f'{outfold}/{file}'):
            return False
    return True


def remove_punctuation_from_sentence(sentence):
    sentence = re.sub('[' + string.punctuation + ']', '', sentence)
    sentence = sentence.lower()
    return sentence


def ids2text(ids, vocab):
    return ' '.join([vocab[int(id_)] for id_ in ids])


def calc_class_weights(label_freq):
    """
    Goal is to give more weight to the classes with less samples
    so as to match the one with the higest frequency. We achieve this by
    dividing the highest frequency by the freq of each label.
    Example -
    [12, 5, 3] -> [12/12, 12/5, 12/3] -> [1, 2.4, 4]

    Here label_freq is assumed to be sorted by the frequency. I.e.
    label_freq[0] is the most frequent element.

    """

    most_common_label_freq = label_freq[0]
    weighted_slots = sorted([(index, most_common_label_freq[1] / freq) for (index, freq) in label_freq])
    return [weight for (_, weight) in weighted_slots]