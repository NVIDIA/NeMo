import torch




def subsequent_mask(size, device="gpu", dtype=torch.uint8):
    """Create mask for subsequent steps (1, size, size)

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)


# def make_pad_mask(lengths, xs=None, length_dim=-1):
#     """Make mask tensor containing indices of padded part.
#     Args:
#         lengths (LongTensor or List): Batch of lengths (B,).
#         xs (Tensor, optional): The reference tensor.
#             If set, masks will be the same shape as this tensor.
#         length_dim (int, optional): Dimension indicator of the above tensor.
#             See the example.
#     Returns:
#         Tensor: Mask tensor containing indices of padded part.
#                 dtype=torch.uint8 in PyTorch 1.2-
#                 dtype=torch.bool in PyTorch 1.2+ (including 1.2)
#     Examples:
#         With only lengths.
#         >>> lengths = [5, 3, 2]
#         >>> make_non_pad_mask(lengths)
#         masks = [[0, 0, 0, 0 ,0],
#                  [0, 0, 0, 1, 1],
#                  [0, 0, 1, 1, 1]]
#         With the reference tensor.
#         >>> xs = torch.zeros((3, 2, 4))
#         >>> make_pad_mask(lengths, xs)
#         tensor([[[0, 0, 0, 0],
#                  [0, 0, 0, 0]],
#                 [[0, 0, 0, 1],
#                  [0, 0, 0, 1]],
#                 [[0, 0, 1, 1],
#                  [0, 0, 1, 1]]], dtype=torch.uint8)
#         >>> xs = torch.zeros((3, 2, 6))
#         >>> make_pad_mask(lengths, xs)
#         tensor([[[0, 0, 0, 0, 0, 1],
#                  [0, 0, 0, 0, 0, 1]],
#                 [[0, 0, 0, 1, 1, 1],
#                  [0, 0, 0, 1, 1, 1]],
#                 [[0, 0, 1, 1, 1, 1],
#                  [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
#         With the reference tensor and dimension indicator.
#         >>> xs = torch.zeros((3, 6, 6))
#         >>> make_pad_mask(lengths, xs, 1)
#         tensor([[[0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [1, 1, 1, 1, 1, 1]],
#                 [[0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1]],
#                 [[0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0],
#                  [1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
#         >>> make_pad_mask(lengths, xs, 2)
#         tensor([[[0, 0, 0, 0, 0, 1],
#                  [0, 0, 0, 0, 0, 1],
#                  [0, 0, 0, 0, 0, 1],
#                  [0, 0, 0, 0, 0, 1],
#                  [0, 0, 0, 0, 0, 1],
#                  [0, 0, 0, 0, 0, 1]],
#                 [[0, 0, 0, 1, 1, 1],
#                  [0, 0, 0, 1, 1, 1],
#                  [0, 0, 0, 1, 1, 1],
#                  [0, 0, 0, 1, 1, 1],
#                  [0, 0, 0, 1, 1, 1],
#                  [0, 0, 0, 1, 1, 1]],
#                 [[0, 0, 1, 1, 1, 1],
#                  [0, 0, 1, 1, 1, 1],
#                  [0, 0, 1, 1, 1, 1],
#                  [0, 0, 1, 1, 1, 1],
#                  [0, 0, 1, 1, 1, 1],
#                  [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
#     """
#     if length_dim == 0:
#         raise ValueError("length_dim cannot be 0: {}".format(length_dim))

#     if not isinstance(lengths, list):
#         lengths = lengths.tolist()
#     bs = int(len(lengths))
#     if xs is None:
#         maxlen = int(max(lengths))
#     else:
#         maxlen = xs.size(length_dim)

#     seq_range = torch.arange(0, maxlen, dtype=torch.int64)
#     seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
#     seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
#     mask = seq_range_expand >= seq_length_expand

#     if xs is not None:
#         assert xs.size(0) == bs, (xs.size(0), bs)

#         if length_dim < 0:
#             length_dim = xs.dim() + lengthdim
#         # ind = (:, None, ..., None, :, , None, ..., None)
#         ind = tuple(
#             slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
#         )
#         mask = mask[ind].expand_as(xs).to(xs.device)
#     return mask


def make_pad_mask(seq_lens, max_time, device=None):
    """Make masking for padding."""
    bs = seq_lens.size(0)
    seq_range = torch.arange(0, max_time, dtype=torch.int32)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_time)
    seq_lens = seq_lens.type(seq_range_expand.dtype).to(seq_range_expand.device)
    seq_length_expand = seq_lens.unsqueeze(-1)
    mask = seq_range_expand < seq_length_expand

    if device:
        mask = mask.to(device)
    return mask


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def add_sos_eos(ys_pad, ignore_id=-1, num=4):
    eos = ys_pad.new([num])  # change this!
    sos = ys_pad.new([num])
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, eos], dim=0) for y in ys]
    return pad_list(ys_in, num), pad_list(ys_out, ignore_id)


def target_mask(ys_in_pad, ignore_id=-1):
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device)
    ys_mask_unsqueeze = ys_mask.type(torch.uint8)

    # logic_and = ys_mask_unsqueeze & m
    # print("logic_and shape", logic_and.shape)
    # return logic_and
    return ys_mask_unsqueeze
