# Evaluation Scripts

This directory contains scripts for deploying and evaluating NeMo models, as well as preparing datasets required for model evaluation. Here, we focus only on the [GPQA main](https://huggingface.co/datasets/Idavidrein/gpqa/viewer/gpqa_main?views%5B%5D=gpqa_main), [GPQA diamond](https://huggingface.co/datasets/Idavidrein/gpqa/viewer/gpqa_diamond), and [MMLU](https://huggingface.co/datasets/cais/mmlu) benchmarks. We will use the datasets hosted on HuggingFace and prepare them using the `prepare_dataset.py` script in the folder. Once the dataset is prepared, we will deploy our trained LoRA checkpoint using the `deploy_and_get_responses.py` script. This script will generate responses for the selected benchmark. Once we have the model responses, we can use the `evaluate_responses.py` script, which compares the ground truth response and extracts the model response.

## Prerequisites

- **Hardware Requirement:** At least 1 GPU is required to run the Llama 8B model. Ensure that your system meets the necessary specifications for GPU usage.
- **Environment Details:** This playbook has been tested on: nvcr.io/nvidia/nemo:25.04. It is expected to work similarly on other environments. Launch the NeMo Framework container as follows:
```bash
docker run -it -p 8080:8080 --rm --gpus '"device=0"' --ipc=host --network host -v $(pwd):/workspace nvcr.io/nvidia/nemo:25.04
```

## Scripts Overview

### 1. `prepare_dataset.py`

**Purpose:** This script is used to prepare a dataset for model evaluation. It accesses one or all of the datasets based on the argument chosen by the user and downloads the benchmark dataset from HuggingFace. The script rearranges the dataset to depict the question, choices, and the correct answer as one of the multiple-choice options ('A', 'B', 'C', 'D').

**How to Run:**
```bash
python prepare_dataset.py --datasets [mmlu, gpqa, gpqa_diamond, all]
```

**Arguments:**
- `--datasets`: Specify which datasets to process. Options are `mmlu`, `gpqa`, `gpqa_diamond`, or `all` (default).

**Step-by-Step:**
1. **Load the Dataset:** The script loads the dataset that you want to prepare.
2. **Process the Dataset:** It processes the dataset to ensure it is in the correct format.
3. **Save the Dataset:** The script saves the processed dataset as jsonl for later use.

**Note:** Please note that [GPQA main](https://huggingface.co/datasets/Idavidrein/gpqa/viewer/gpqa_main?views%5B%5D=gpqa_main), [GPQA diamond](https://huggingface.co/datasets/Idavidrein/gpqa/viewer/gpqa_diamond) benchmarks are gated repo. In order to clone these login to huggingface cli and enter your token.
```bash
huggingface-cli login
```

### 2. `deploy_and_get_responses.py`

**Purpose:** First, you need to prepare a NeMo 2 checkpoint of the model you would like to evaluate. Assuming we will be using a NeMo2 checkpoint we have trained in the previous step, make sure to mount the directory containing the checkpoint when starting the container. The script below will start a server for the provided checkpoint in a separate process. The script will deploy the model using the Triton Inference Server and set up OpenAI-like endpoints for querying it. The server exposes three endpoints:

- `/v1/triton_health`
- `/v1/completions/`
- `/v1/chat/completions/`

The `/v1/triton_health` allows you to check if the underlying Triton server is ready. The `/v1/completions/` endpoint allows you to send a prompt to the model as-is, without applying the chat template. The model responds with a text completion. Finally, the `/v1/chat/completions/` endpoint allows for multi-turn conversational interactions with the model. This endpoint accepts a structured list of messages with different roles (system, user, assistant) to maintain context and generates chat-like responses. Under the hood, a chat template is applied to turn the conversation into a single input string.

**Note:** Please note that the chat endpoint will not work correctly for base models, as they do not define a chat template. Deployment can take a couple of minutes, especially for larger models.

**How to Run:**
```bash
python deploy_and_get_responses.py --checkpoint_path <checkpoint_path> --dataset <dataset> --output_prefix <output_prefix> --max_tokens <max_tokens>
```

**Arguments:**
- `--checkpoint_path`: Path to the model checkpoint.
- `--dataset`: Dataset to evaluate on (choices: `gpqa_main`, `mmlu`, `gpqa_diamond`).
- `--output_prefix`: Prefix for the output file name (default: `evaluation_results`).
- `--max_tokens`: Maximum number of tokens to generate.

**Step-by-Step:**
1. **Load the Nemo Model:** The script loads the Nemo model that you want to deploy.
2. **Configure Deployment:** It configures the deployment settings. You can modify the caht template here to provide 'detailed thinking on or off' for your reasoining model.
3. **Deploy the Model:** The script deploys the model and outputs the deployment status.

### 3. `evaluate_responses.py`

**Purpose:** This script will ingest the model responses generated in the previous step and perform extraction of the final answer from the model response. Once we extract the model response, we compare it with the ground truth and evaluate the model's performance.

**How to Run:**
```bash
python evaluate_responses.py --input_file <input_file> --output_file <output_file> --model_name <model_name>
```

**Arguments:**
- `--input_file`: Path to the input JSONL file containing model responses.
- `--output_file`: Path to the output CSV file.
- `--model_name`: Name of the model for reporting results.
