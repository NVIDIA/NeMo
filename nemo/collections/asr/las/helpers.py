from itertools import chain
from pprint import pformat

import torch

from nemo.backends.pytorch.common.metrics import char_lm_metrics
from nemo.collections.asr.metrics import word_error_rate
from nemo.utils import logging

ENG_MWN = 5.3


def process_evaluation_batch(tensors, global_vars, labels, specials, tb_writer=None, write_attn=True):
    loss, log_probs = ([],) * 2
    transcripts, transcript_texts = ([],) * 2
    predictions, prediction_texts = ([],) * 2
    attention_weights = []
    for k, v in tensors.items():
        if 'loss' in k:
            loss = v
        elif 'log_probs' in k:
            log_probs = v
        elif ('transcripts' in k) or ('texts' in k):
            transcripts = v
            transcript_texts = __decode(v, labels, specials)
        elif 'predictions' in k:
            # predictions = v
            prediction_texts = __decode(v, labels, specials)
        elif 'attention_weights' in k:
            attention_weights = v

    global_vars.setdefault('loss', [])
    global_vars['loss'].extend(loss)
    bpc, ppl = char_lm_metrics(log_probs, transcripts, transcript_texts, specials['pad_id'])
    global_vars.setdefault('bpc', [])
    global_vars['bpc'].extend(bpc)
    global_vars.setdefault('ppl', [])
    global_vars['ppl'].extend(ppl)
    global_vars.setdefault('transcript_texts', [])
    global_vars['transcript_texts'].extend(transcript_texts)
    global_vars.setdefault('prediction_texts', [])
    global_vars['prediction_texts'].extend(prediction_texts)

    # TODO: Add step number?
    if tb_writer is not None and len(attention_weights) and write_attn:
        sample_len = len(prediction_texts[0][0])
        if sample_len > 0:
            attention_weights = attention_weights[0][0, :sample_len, :]
            tb_writer.add_image(
                'image/eval_attention_weights', attention_weights, dataformats='HW',
            )


def process_evaluation_epoch(
    global_vars, metrics=('loss', 'bpc', 'ppl'), calc_wer=False, mode='eval', tag='none',
):
    tag = '_'.join(tag.lower().strip().split())
    return_dict = {}
    for metric in metrics:
        value = torch.mean(torch.stack(global_vars[metric])).item()
        return_dict[f'metric/{mode}_{metric}_{tag}'] = value

    # TODO: Delete?
    bpc = return_dict[f'metric/{mode}_bpc_{tag}']
    return_dict[f'metric/{mode}_ppl_{tag}'] = 2 ** (bpc * ENG_MWN)

    if calc_wer:
        transcript_texts = list(chain(*global_vars['transcript_texts']))
        prediction_texts = list(chain(*global_vars['prediction_texts']))

        logging.info(f'Ten examples (transcripts and predictions)')
        logging.info(transcript_texts[:10])
        logging.info(prediction_texts[:10])

        wer = word_error_rate(hypotheses=prediction_texts, references=transcript_texts)
        return_dict[f'metric/{mode}_wer_{tag}'] = wer

    logging.info(pformat(return_dict))

    return return_dict


def __decode(tensors_list, labels, specials):
    labels_map = dict([(i, labels[i]) for i in range(len(labels)) if i not in set(specials.values())])
    results = []
    for tensor in tensors_list:
        tensor = tensor.long().cpu()
        hypotheses = []
        for i in range(tensor.shape[0]):
            hypothesis = ''.join([labels_map[c] for c in tensor[i].numpy().tolist() if c in labels_map])
            hypotheses.append(hypothesis)

        results.append(hypotheses)

    return results
