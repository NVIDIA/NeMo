from nemo.collections.nlp.data import TranslationDataset


def test_pack_data_into_batches():
    tokens_in_batch = 17
    dataset = TranslationDataset("", "", tokens_in_batch=tokens_in_batch)
    dataset.src_pad_id = 0
    dataset.tgt_pad_id = 0
    src = [[1, 2, 3]] * 9
    tgt = [[1, 2, 3]] * 9
    batches = dataset.pack_data_into_batches(src, tgt)
    padded_batch = dataset.pad_batches(src, tgt, batches)
    for i, batch in padded_batch.items():
        src_batch_size = batch['src'].size
        tgt_batch_size = batch['tgt'].size
        assert src_batch_size + tgt_batch_size <= tokens_in_batch