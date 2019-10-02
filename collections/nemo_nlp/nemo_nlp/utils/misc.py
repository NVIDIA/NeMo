def mask_padded_tokens(tokens, pad_id):
    mask = (tokens != pad_id)
    return mask
