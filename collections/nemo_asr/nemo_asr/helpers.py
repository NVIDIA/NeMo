# Copyright (c) 2019 NVIDIA Corporation

import torch

from .metrics import word_error_rate


def __ctc_decoder_predictions_tensor(tensor, labels):
    """
    Decodes a sequence of labels to words
    """
    blank_id = len(labels)
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for ind in range(prediction_cpu_tensor.shape[0]):
        prediction = prediction_cpu_tensor[ind].numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = len(labels)  # id of a blank symbol
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = ''.join([labels_map[c] for c in decoded_prediction])
        hypotheses.append(hypothesis)
    return hypotheses


def monitor_asr_train_progress(tensors: list, labels: list, tb_logger=None,
                               logger=None):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Prints sample to screen, computes
    and logs AVG WER to console and (optionally) Tensorboard
    Args:
      tensors: A list of 3 tensors (predictions, targets, target_lengths)
      labels: A list of labels
      tb_logger: Tensorboard logging object

    Returns:
      None
    """
    references = []

    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    with torch.no_grad():
        # prediction_cpu_tensor = tensors[0].long().cpu()
        targets_cpu_tensor = tensors[2].long().cpu()
        tgt_lenths_cpu_tensor = tensors[3].long().cpu()

        # iterate over batch
        for ind in range(targets_cpu_tensor.shape[0]):
            tgt_len = tgt_lenths_cpu_tensor[ind].item()
            target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
            reference = ''.join([labels_map[c] for c in target])
            references.append(reference)
        hypotheses = __ctc_decoder_predictions_tensor(
            tensors[1], labels=labels)
    tag = "training_batch_WER"
    wer = word_error_rate(hypotheses, references)
    if tb_logger is not None:
        tb_logger.add_scalar(tag, wer)
    if logger:
        logger.info(f'Loss: {tensors[0]}')
        logger.info(f'{tag}: {wer*100 : 5.2f}%')
        logger.info(f'Prediction: {hypotheses[0]}')
        logger.info(f'Reference: {references[0]}')
    else:
        print(f'Loss: {tensors[0]}')
        print(f'{tag}: {wer*100 : 5.2f}%')
        print(f'Prediction: {hypotheses[0]}')
        print(f'Reference: {references[0]}')


def __gather_losses(losses_list: list) -> list:
    return [torch.mean(torch.stack(losses_list))]


def __gather_predictions(predictions_list: list, labels: list) -> list:
    results = []
    for prediction in predictions_list:
        results += __ctc_decoder_predictions_tensor(prediction, labels=labels)
    return results


def __gather_transcripts(transcript_list: list, transcript_len_list: list,
                         labels: list) -> list:
    results = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    # iterate over workers
    for t, ln in zip(transcript_list, transcript_len_list):
        # iterate over batch
        t_lc = t.long().cpu()
        ln_lc = ln.long().cpu()
        for ind in range(t.shape[0]):
            tgt_len = ln_lc[ind].item()
            target = t_lc[ind][:tgt_len].numpy().tolist()
            reference = ''.join([labels_map[c] for c in target])
            results.append(reference)
    return results


def process_evaluation_batch(tensors: dict, global_vars: dict, labels: list):
    """
    Creates a dictionary holding the results from a batch of audio
    """
    if 'EvalLoss' not in global_vars.keys():
        global_vars['EvalLoss'] = []
    if 'predictions' not in global_vars.keys():
        global_vars['predictions'] = []
    if 'transcripts' not in global_vars.keys():
        global_vars['transcripts'] = []
    if 'logits' not in global_vars.keys():
        global_vars['logits'] = []
    # if not 'transcript_lengths' in global_vars.keys():
    #  global_vars['transcript_lengths'] = []
    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += __gather_losses(v)
        elif kv.startswith('predictions'):
            global_vars['predictions'] += __gather_predictions(
                v, labels=labels)
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            global_vars['logits'] += v

    global_vars['transcripts'] += __gather_transcripts(transcript_list,
                                                       transcript_len_list,
                                                       labels=labels)


def process_evaluation_epoch(global_vars: dict, tag=None, logger=None):
    """
    Calculates the aggregated loss and WER across the entire evaluation dataset
    """
    eloss = torch.mean(torch.stack(global_vars['EvalLoss'])).item()
    hypotheses = global_vars['predictions']
    references = global_vars['transcripts']

    wer = word_error_rate(hypotheses=hypotheses, references=references)
    if tag is None:
        if logger:
            logger.info(f"==========>>>>>>Evaluation Loss: {eloss}")
            logger.info(f"==========>>>>>>Evaluation WER: {wer*100 : 5.2f}%")
        else:
            print(f"==========>>>>>>Evaluation Loss: {eloss}")
            print(f"==========>>>>>>Evaluation WER: {wer*100 : 5.2f}%")
        return {"Evaluation_Loss": eloss, "Evaluation_WER": wer}
    else:
        if logger:
            logger.info(f"==========>>>>>>Evaluation Loss {tag}: {eloss}")
            logger.info(f"==========>>>>>>Evaluation WER {tag}: "
                        f"{wer*100 : 5.2f}%")
        else:
            print(f"==========>>>>>>Evaluation Loss {tag}: {eloss}")
            print(f"==========>>>>>>Evaluation WER {tag}: {wer*100 : 5.2f}%")
        return {f"Evaluation_Loss_{tag}": eloss, f"Evaluation_WER_{tag}": wer}


def post_process_predictions(predictions, labels):
    return __gather_predictions(predictions, labels=labels)


def post_process_transcripts(
        transcript_list, transcript_len_list, labels):
    return __gather_transcripts(transcript_list,
                                transcript_len_list,
                                labels=labels)
