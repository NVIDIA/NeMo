import pytest

from nemo.collections.nlp.data import TranslationDataset


@pytest.mark.parametrize(
    "segment_length, num_segments, tokens_in_batch, num_expected_batches",
    [(4096, 2, 8192, 2), (512, 8, 8192, 1), (512, 9, 8192, 2), (1, 8192, 8192, 2),],
)
def test_pack_data_into_batches(segment_length, num_segments, tokens_in_batch, num_expected_batches):
    dataset = TranslationDataset("", "", tokens_in_batch=tokens_in_batch)
    dataset.src_pad_id = 0
    dataset.tgt_pad_id = 0
    src = [[1 for _ in range(segment_length)]] * num_segments
    tgt = [[1 for _ in range(segment_length)]] * num_segments
    batches = dataset.pack_data_into_batches(src, tgt)
    assert len(batches) == num_expected_batches
    padded_batch = dataset.pad_batches(src, tgt, batches)
    for i, batch in padded_batch.items():
        src_batch_size = batch['src'].size
        tgt_batch_size = batch['tgt'].size
        assert src_batch_size + tgt_batch_size <= tokens_in_batch
