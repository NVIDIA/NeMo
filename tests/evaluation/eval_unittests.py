from unittest.mock import patch

import pytest

from nemo.collections.llm.api import evaluate
from nemo.collections.llm.evaluation.api import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget


@pytest.mark.parametrize(
    "params",
    [
        {
            "top_p": 0.1,
            "temperature": 0.001,
        },
        {"limit_samples": 10},
        {"limit_samples": 0.1},
        {"max_new_tokens": 64},
        {"max_retries": 10, "parallelism": 16, "request_timeout": 100},
        {"task": "my_task", "extra": {"num_fewshot": 5, "tokenizer": "my_tokenizer"}},
    ],
)
def test_configuration(params: dict):
    eval_config = EvaluationConfig(type="custom", params=params)
    assert isinstance(eval_config.params, ConfigParams)
    assert eval_config.type == "custom"
    for param_name, param_value in params.items():
        assert getattr(eval_config.params, param_name) == param_value


def test_default_none_tokenizer():
    eval_config = EvaluationConfig(type="custom", params={"extra": {"num_fewshot": 5}})
    assert eval_config.type == "custom"
    assert eval_config.params.extra["tokenizer"] is None
    assert eval_config.params.extra["num_fewshot"] == 5


def test_evaluation():
    target_config = EvaluationTarget(api_endpoint={"url": "http://my_endpoint/v1/completions/", "type": "completions"})
    eval_config = EvaluationConfig(
        type="gsm8k",
        params=ConfigParams(
            extra={"tokenizer": "Qwen/Qwen2.5-0.5B", "num_fewshot": 13}, limit_samples=1, parallelism=1
        ),
    )
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200
        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'id': 'cmpl-123456',
            'object': 'text_completion',
            'created': 1234567,
            'model': 'triton_model',
            'choices': [
                {
                    'text': ' Janet eats 3 eggs and bakes 4 eggs, so she has 16 - 3 - 4 = <<16-3-4=9>>9 eggs left.\nShe sells 9 eggs for $2 each, so she makes 9 x 2 = <<9*2=18>>18 dollars.\n#### 18'
                }
            ],
        }

        results = evaluate(target_cfg=target_config, eval_cfg=eval_config)
        assert results is not None
