from typing import Optional

from pydantic import BaseModel, Field


class ApiEndpoint(BaseModel):
    url: str = Field(description="Url of the model")
    model_id: str = Field(description="Name of the model")
    type: str = Field(description="The type of the target", default="chat")
    api_key: str = Field(
        description="Name of the env variable that stores API key for the model", default="NVIDIA_API_KEY"
    )
    stream: bool = Field(description="Whether responses should be streamed", default=False)
    nemo_checkpoint_path: Optional[str] = Field(
        description="Path for nemo 2.0 checkpoint. This is used to get the tokenizer from the ckpt which is required to tokenize the evaluation input and output prompts.",
        default=None,
    )
    nemo_triton_http_port: Optional[int] = Field(
        description="HTTP port that was used for the PyTriton server in the deploy method. Default: 8000.",
        default=8000,
    )


class EvaluationTarget(BaseModel):
    api_endpoint: ApiEndpoint = Field(description="Api endpoint to be used for evaluation")


class ConfigParams(BaseModel):
    parallelism: int = Field(description="Parallelism to be used", default=1)
    top_p: float = Field(
        description="float value between 0 and 1. limits to the top tokens within a certain probability. top_p=0 means the model will only consider the single most likely token for the next prediction. Default: 0.9999999",
        default=0.9999999,
    )
    temperature: float = Field(
        description="float value between 0 and 1. temp of 0 indicates greedy decoding, where the token with highest prob is chosen. Temperature can't be set to 0.0 currently. Default: 0.0000001",
        default=0.0000001,
    )
    tokenizer_path: str = Field(
        description="Name of the tokenizer used for evaluation", default="meta-llama/Llama-3.1-70B-Instruct"
    )
    limit: Optional[int] = Field(
        description="Limit evaluation to `limit` samples. Default: use all samples", default=None
    )
    first_n: Optional[int] = Field(
        description="Evaluate only on the first first_n samples. Default: use all samples", default=None
    )
    n_samples: Optional[int] = Field(description="Number of samples to be generated", default=1)
    num_samples: Optional[int] = Field(description="Maximum number of samples to test (in ruler)", default=10)
    num_fewshot: Optional[int] = Field(
        description="Number of examples in few-shot context. Default: None.", default=None
    )
    max_tokens_to_generate: Optional[int] = Field(description="max tokens to generate. Default: 256.", default=256)
    top_k: Optional[int] = Field(
        description="limits to a certain number (K) of the top tokens to consider. top_k=1 means the model will only consider the single most likely token for the next prediction. Default: 1",
        default=1,
    )
    add_bos: Optional[bool] = Field(
        description="whether a special token representing the beginning of a sequence should be added when encoding a string. Default: False.",
        default=False,
    )
    bootstrap_iters: int = Field(
        description="Number of iterations for bootstrap statistics, used when calculating stderrs. Set to 0 for no stderr calculations to be performed. Default: 100000.",
        default=100000,
    )


class EvaluationConfig(BaseModel):
    type: str = Field(description="Name/type of the task")
    params: ConfigParams = Field(description="Parameters to be used for evaluation", default=ConfigParams())
