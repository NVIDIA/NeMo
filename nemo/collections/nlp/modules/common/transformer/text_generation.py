from typing import List, TypedDict, Union

from torch import Tensor


class LengthParam(TypedDict):
    max_length: int  # The maximum length of the sequence to be generated.
    min_length: int  # The minimum length of the sequence to be generated.


class SamplingParam(TypedDict):
    use_greedy: bool  # Whether or not to use sampling ; use greedy decoding otherwise
    top_k: int  # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_p: float  # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    repetition_penalty: float  # The parameter for repetition penalty. 1.0 means no penalty.
    add_BOS: bool  # add the bos token at the begining of the prompt


class OutputType(TypedDict):
    sentences: List[str]  # output sentences
    tokens: List[List[str]]  # output sentences borken into tokens
    logits: List[Tensor]  # logits output for all the tokens
    token_ids: List[Tensor]  # output sentence token ids


class TextGeneration:
    """
    Interface for all text generation models.
    """

    def generate(self, inputs: Union[List[str], Tensor, List[dict]], length_params: LengthParam, sampling_params: SamplingParam = None) -> OutputType:
        """
        Public method to generate text.

        Args:
            inputs (Union[List[str], Tensor, List[dict]]):
                Can be one of the 3 types: 
                1. List of strings. Each element of the list provides input prompt. The model will apply tokenizer on it.
                    E.g [‘sentence’, ‘sentence2’ … ]
                2. Pytorch Tensor of shape (batch_size, seq_length).  The sequence used as a prompt for the generation or as model inputs to the encoder. It will skip the tokenization step.
                    E.g. torch.tensor([[23,5234,23,35…], [223,323,23,23232,232] …])    
                3. List of python dict objects. Used for prompt/p-tuning inputs where a set of key-value pairs are converted into input token embeddings for the model.
                    E.g. [{"prompt-tag": "sentiment", "sentence": "this is a good movie"},
                          {"prompt-tag": "qa", "context": "some context text", "question": "a simple question"} ... ]
                          where 'prompt-tag' is used to identify the type of NLP task to solve.
            length_params (LengthParam):
                a dictionary type which controls the sampling length.
                    max_length: int, The maximum length of the sequence to be generated.
                    min_length: int,  The minimum length of the sequence to be generated.
                If None, max_length is set to 30, and min_length is set to None
            sampling_params (SamplingParam):
                a dictionary type which contains the parameters for text sampling. It has the following keys
                    use_greedy: bool,  Whether or not to use sampling ; use greedy decoding otherwise
                    top_k: int, The number of highest probability vocabulary tokens to keep for top-k-filtering.
                    top_p: float, If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                    repetition_penalty: float, The parameter for repetition penalty. 1.0 means no penalty. 
                    add_BOS: bool, Whether add the bos token at the begining of the prompt
                Default None, If it is None, use_greedy will be "True".
        Returns:
            OutputType: It generates the output in a dictionary type. It has the following keys:
                sentences: List[str], output sentences
                tokens: List[List[str]], output sentences borken into tokens
                logits: List[Tensor], logits output for all the tokens
                token_ids: List[Tensor], output sentence token ids
        """
        raise NotImplementedError("please implement this method")