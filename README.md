# Half-duplex AVLM

- [More details](https://docs.google.com/document/d/1k8PAEgl-Fe-hakWW20lH2JAR6iVUHsWrvmRcZxVYmyQ/edit?usp=sharing)

## Authors

- **Yuanhang Su** (US)
- **Ehsan Hosseini Asl** (US)
- **Danial Mohseni Taheri** (US)
- **Leili Tavabi** (US)

## Cosmos Nemotron (VILA)

### Data

**Dataset for VILA:** Commercial VILA Comparison\
**Nemo 2.0:** [NeMo Model](https://github.com/NVIDIA/NeMo/blob/0eb9e5d3367ffec47650df17ae89b3e5be6eaa41/nemo/collections/multimodal/models/multimodal_llm/neva/neva_model.py)

### LLaVA-NeXT

**Script:** [LLaVA-NeXT Pretrain Script](https://github.com/NVIDIA/NeMo/blob/main/scripts/vlm/llava_next_pretrain.py)

## Nemoaudio

### Data

**ASR/AST/SQA/Audio Understanding**

- **AudioLM Datasets**
- **SFT blend (brainy mantis)** – Synthesized audio inputs for the user question
- **Original text version:**
  ```
  draco-oci: /lustre/fsw/portfolios/llmservice/users/ameyasunilm/datasets/Minitron-SFT/brainymantis+jsontoolcalling20k+jailbreak15k_minitron-id.jsonl
  ```

## Nemo 1.0

- **Codebase:** [NeMo Speech LLM](https://github.com/Leili/NeMo/tree/speechllm-develop-prompt)
- **Training Script:**
  ```
  cs-oci:/lustre/fsw/portfolios/llmservice/users/ltavabi/audiollm/scripts/mn-minitron-v3/sft_train_twostage_stage1_v3_multiturn.sh
  ```
- **Inference Script:**
  ```
  cs-oci:/lustre/fsw/portfolios/llmservice/users/ltavabi/audiollm/scripts/mn-minitron-v3/mn_inference.sh
  ```

## Nemo 2.0

- **Codebase:** [NeMo Speech-to-Text LLM](https://github.com/NVIDIA/NeMo/blob/nemo/collections/speechlm/models/speech_to_text_llm_model.py#L533)
- **Training Script:** [Speech-to-Text LLM Train Script](https://github.com/NVIDIA/NeMo/blob/main/examples/speechlm/speech_to_text_llm_train.py)
- **Config:** [SpeechLM Configurations](https://github.com/NVIDIA/NeMo/tree/main/examples/speechlm/conf/salm)

## AVLM

### Data

- **AVLM Datasets**


