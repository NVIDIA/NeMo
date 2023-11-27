import cudf.pandas

cudf.pandas.install()
import pandas as pd


class cuDF:
    def __init__(self):
        pass

    def load_samples_chunk(self, samples: list[dict]):
        chunk = pd.DataFrame(samples)
        return chunk

    def concat_samples_chunks(self, samples_chunks: list):
        samples_datatable = pd.concat(samples_chunks).reset_index(drop=True)
        return samples_datatable

    def process_vocabulary(self, words_frequencies: dict, hypotheses_metrics: list[object]):
        vocabulary_dfs = []

        words_frequencies_df = pd.DataFrame(words_frequencies.items(), columns=["Word", "Amount"]).set_index("Word")
        vocabulary_dfs.append(words_frequencies_df)

        for hypothesis_metrics_obj in hypotheses_metrics:
            label = hypothesis_metrics_obj.hypothesis_label
            match_words_frequencies = hypothesis_metrics_obj.match_words_frequencies
            match_words_frequencies_df = pd.DataFrame(
                match_words_frequencies.items(), columns=["Word", f"Match_{hypothesis_metrics_obj.hypothesis_label}"]
            ).set_index("Word")
            vocabulary_dfs.append(match_words_frequencies_df)

        vocabulary_datatable = pd.concat(vocabulary_dfs, axis=1, join="outer").reset_index().fillna(0)

        for hypothesis_metrics_obj in hypotheses_metrics:
            label = hypothesis_metrics_obj.hypothesis_label
            postfix = ""
            if label != "":
                postfix = f"_{label}"

            vocabulary_datatable[f"Accuracy{postfix}"] = (
                vocabulary_datatable[f"Match_{label}"] / vocabulary_datatable["Amount"] * 100
            )
            vocabulary_datatable[f"Accuracy{postfix}"] = vocabulary_datatable[f"Accuracy{postfix}"].round(2)
            vocabulary_datatable = vocabulary_datatable.drop(f"Match_{label}", axis=1)
            hypothesis_metrics_obj.mwa = round(vocabulary_datatable[f"Accuracy{postfix}"].mean(), 2)

        return vocabulary_datatable
