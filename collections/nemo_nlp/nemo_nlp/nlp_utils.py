import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from nemo.utils.exp_logging import get_logger

logger = get_logger('')


def read_intent_slot_outputs(queries,
                             intent_file,
                             slot_file,
                             intent_logits,
                             slot_logits,
                             slot_masks,
                             intents=None,
                             slots=None):
    intent_dict = get_vocab(intent_file)
    slot_dict = get_vocab(slot_file)
    pred_intents = np.argmax(intent_logits, 1)
    pred_slots = np.argmax(slot_logits, axis=2)
    for i, query in enumerate(queries):
        logger.info(f'Query: {query}')
        pred = pred_intents[i]
        logger.info(f'Predicted intent:\t{pred}\t{intent_dict[pred]}')
        if intents is not None:
            logger.info(
                f'True intent:\t{intents[i]}\t{intent_dict[intents[i]]}')

        pred_slot = pred_slots[i][slot_masks[i]][1:-1]
        tokens = query.strip().split()

        if len(pred_slot) != len(tokens):
            raise ValueError('Pred_slot and tokens must be of the same length')

        for j, token in enumerate(tokens):
            output = f'{token}\t{slot_dict[pred_slot[j]]}'
            if slots is not None:
                output = f'{output}\t{slot_dict[slots[i][j]]}'
            logger.info(output)


def get_vocab(file):
    lines = open(file, 'r').readlines()
    labels = {i: lines[i].strip() for i in range(len(lines))}
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


def write_vocab_in_order(vocab, outfile):
    with open(outfile, 'w') as f:
        for key in sorted(vocab.keys()):
            f.write(f'{vocab[key]}\n')
