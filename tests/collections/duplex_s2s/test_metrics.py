from nemo.collections.duplex_s2s.parts.metrics import BLEU


def test_bleu():
    metric = BLEU(verbose=False)
    metric.update(
        name="dataset_1",
        refs=["a b c d e f g h i j k l", "m n o p r s t u v"],
        hyps=["a b c d e f g h i j k l", "m n o p r s t u v"],
    )
    metric.update(
        name="dataset_2",
        refs=["a b c"],
        hyps=["a b d"],
    )
    ans = metric.compute()
    assert ans["txt_bleu_dataset_1"] == 100.0
    assert ans["txt_bleu_dataset_2"] == 0.0
    assert ans["txt_bleu"] == 50.0  # average across datasets
