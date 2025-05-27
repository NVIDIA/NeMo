# Evaluation Adapters

Evaluation adapters provide a flexible mechanism for intercepting and modifying requests/responses between the evaluation harness and the model endpoint. This allows for custom processing, logging, and transformation of data during the evaluation process.

## Architecture

The adapter system uses a chain of interceptors that process requests and responses in sequence. Here's the high-level architecture:

```
         ┌───────────────────────┐
         │                       │
         │ NVIDIA Eval Factory   │
         │                       │
         └───▲──────┬────────────┘
             │      │
     returns │      │
             │      │ calls
             │      │
             │      │
         ┌───┼──────┼──────────────────────────────────────────────────┐
         │   │      ▼                                                  │
         │ AdapterServer (@ localhost:<free port>)                     │
         │                                                             │
         │   ▲      │       chain of RequestInterceptors:              │
         │   │      │       flask.Request                              │
         │   │      │       is passed on the way up                    │
         │   │      │                                                  │   ┌──────────────────────┐
         │   │ ┌────▼───────────────────────────────────────────────┐  │   │                      │
         │   │ │intcptr_1─────►intcptr_2───►...───►intcptr_N────────┼──┼───►                      │
         │   │ │                                                    │  │   │                      │
         │   │ └────────────────────────────────────────────────────┘  │   │                      │
         │   │                                                         │   │  upstream endpoint   │
         │   │                                                         │   │   with actual model  │
         │   │                                                         │   │                      │
         │   │                                                         │   │                      │
         │   │                                                         │   │                      │
         │ ┌─┼──────────────────────────────────────────┐              │   │                      │
         │ │intcptr'_M◄──intcptr'_2◄──...◄───intcptr'_1 ◄──────────────┼───┤                      │
         │ └────────────────────────────────────────────┘              │   └──────────────────────┘
         │                                                             │
         │              Chain of ResponseInterceptors:                 │
         │              requests.Response is passed on the way down    │
         │                                                             │
         │                                                             │
         └─────────────────────────────────────────────────────────────┘
```

The adapter server runs locally and acts as a proxy between the evaluation harness and the model endpoint. It processes requests through a chain of request interceptors before forwarding them to the endpoint, and then processes responses through a chain of response interceptors before returning them to the harness.

## Configuration

The adapter system is configured using the `AdapterConfig` class with the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `local_port` | `Optional[int]` | `None` | Local port to use for the adapter server. If `None`, a free port will be chosen automatically |
| `use_reasoning` | `bool` | `False` | Whether to use the clean-reasoning-tokens adapter |
| `end_reasoning_token` | `str` | `"</think>"` | Token that signifies the end of reasoning output |
| `custom_system_prompt` | `Optional[str]` | `None` | A custom system prompt to replace the original one (if not None), **only for chat endpoints** |
| `max_logged_responses` | `Optional[int]` | `5` | Maximum number of responses to log. Set to 0 to disable. If None, all will be logged |
| `max_logged_requests` | `Optional[int]` | `5` | Maximum number of requests to log. Set to 0 to disable. If None, all will be logged |


## Usage Example

To enable the adapter server, pass `AdapterConfig` class to the `evaluate`  method of the `nemo/collections/llm/api.py`.
Taking an example from `tutorials/llm/evaluation/mmlu.ipynb`, we can modify it

```python
target_config = EvaluationTarget(api_endpoint={"url": completions_url, "type": "completions"})
eval_config = EvaluationConfig(
    type="mmlu",
    params={"limit_samples": 1},
    output_dir=f"{WORKSPACE}/mmlu",
)
adapter_config = AdapterConfig(
    local_port=None,
    max_logged_requests=1,
    max_logged_responses=1,
)

results = api.evaluate(
    target_cfg=target_config,
    eval_cfg=eval_config,
    adapter_cfg=adapter_config,
)
```

