import cudf.pandas

cudf.pandas.install()
import pandas as pd


class CuDFEngine:
    """
    A class representing a data engine utilizing cuDF for processing and analysis.

    Methods:
    - load_samples_chunk(samples: list[dict]) -> pd.DataFrame:
        Loads a chunk of samples into a cuDF DataFrame.

    - concat_samples_chunks(samples_chunks: list) -> pd.DataFrame:
        Concatenates a list of sample chunks into a single cuDF DataFrame.

    - process_vocabulary(words_frequencies: dict, hypotheses_metrics: list[object]) -> pd.DataFrame:
        Processes vocabulary information, including word frequencies and match frequencies for hypotheses.

    """

    def __init__(self):
        pass

    def load_samples_chunk(self, samples: list[dict]):
        """
        Loads a chunk of samples into a cuDF DataFrame.

        Parameters:
        - samples (list[dict]): List of dictionaries representing samples.

        Returns:
        - pd.DataFrame: cuDF DataFrame containing the loaded samples.
        """

        chunk = pd.DataFrame(samples)
        return chunk

    def concat_samples_chunks(self, samples_chunks: list):
        """
        Concatenates a list of sample chunks into a single cuDF DataFrame.

        Parameters:
        - samples_chunks (list): List of cuDF DataFrames representing sample chunks.

        Returns:
        - pd.DataFrame: Concatenated cuDF DataFrame containing all samples.
        """

        samples_datatable = pd.concat(samples_chunks).reset_index(drop=True)
        return samples_datatable

    def process_vocabulary(self, words_frequencies: dict, hypotheses_metrics: list[object]):
        """
        Processes vocabulary information, including word frequencies and match frequencies for hypotheses.

        Parameters:
        - words_frequencies (dict): Dictionary containing word frequencies.
        - hypotheses_metrics (list[object]): List of HypothesisMetrics objects representing hypotheses.

        Returns:
        - pd.DataFrame: Processed vocabulary datatable containing word and match frequencies.
        """

        vocabulary_datatable = pd.DataFrame(words_frequencies.items(), columns=["Word", "Amount"]).set_index("Word")

        for hypothesis_metrics_obj in hypotheses_metrics:
            label = hypothesis_metrics_obj.hypothesis_label
            postfix = ""
            if label != "":
                postfix = f"_{label}"
            
            match_words_frequencies = hypothesis_metrics_obj.match_words_frequencies
            match_words_frequencies_df = pd.DataFrame(
                match_words_frequencies.items(), columns=["Word", f"Match_{postfix}"]
            ).set_index("Word")
            
            vocabulary_datatable = pd.concat([vocabulary_datatable, match_words_frequencies_df], axis = 1, join="outer").fillna(0)
            vocabulary_datatable[f"Accuracy{postfix}"] = (
                vocabulary_datatable[f"Match_{label}"] / vocabulary_datatable["Amount"] * 100
            ).round(2)
            vocabulary_datatable = vocabulary_datatable.drop(f"Match_{label}", axis=1)
            
            hypothesis_metrics_obj.mwa = round(vocabulary_datatable[f"Accuracy{postfix}"].mean(), 2)

        vocabulary_datatable = vocabulary_datatable.reset_index()
        
        return vocabulary_datatable
