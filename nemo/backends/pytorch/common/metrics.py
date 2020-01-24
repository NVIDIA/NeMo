import torch


def char_lm_metrics(
    chars_log_probs_batches, chars_targets_batches, targets_texts_batches, pad_id,
):
    """Calculate metrics for language modeling.

    Args:
        chars_log_probs_batches:
        chars_targets_batches:
        targets_texts_batches:
        pad_id:

    Returns:
        A tuple of two:
            bpc
            perplexity

    """

    bpcs, ppls = [], []
    for log_probs, targets, texts in zip(chars_log_probs_batches, chars_targets_batches, targets_texts_batches):
        target_log_probs = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
        pad_mask = (targets != pad_id).long()
        nll = -(target_log_probs * pad_mask.float()).sum(-1)
        char_lens = pad_mask.float().sum(-1)
        word_lens = torch.tensor([len(text.split()) for text in texts], dtype=torch.float, device=char_lens.device,)
        bpc = nll / char_lens
        ppl = 2 ** (nll / word_lens)
        # ppl = 2 ** (bpc * ENG_MWN)  # ~5.3
        bpcs.append(bpc)
        ppls.append(ppl)

    bpc, ppl = torch.cat(bpcs), torch.cat(ppls)

    return bpc, ppl
