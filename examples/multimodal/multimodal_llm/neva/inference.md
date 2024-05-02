## Inference with multimodal

We can run `neva_evaluation.py` to generate inference result from video neva model.
Currently video neva supports both image and video inference by changing the config attribute inference.media_type in `conf/video_neva_inference.yaml` to either `image` or `video`, and add conrresponding images path `inference.images_base_path` or videos path `inference.videos_base_path`.

### inference with pretrained projectors with base LM model

An example of an inference script execution:

For running video inference:

```
CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /path/to/neva_evaluation.py \
--config-path=/path/to/conf/ \
--config-name=video_neva_inference.yaml
tensor_model_parallel_size=4 \
pipeline_model_parallel_size=1 \
neva_model_file=/path/to/projector/checkpoint \
base_model_file=/path/to/base/lm/checkpoint \
trainer.devices=4 \
trainer.precision=bf16 \
prompt_file=/path/to/prompt/file \
inference.videos_base_path=/path/to/videos \
inference.media_type=video \
output_file=/path/for/output/file/ \
inference.temperature=0.2 \
inference.top_k=0 \
inference.top_p=0.9 \
inference.greedy=False \
inference.add_BOS=False \
inference.all_probs=False \
inference.repetition_penalty=1.2 \
inference.insert_media_token=right \
inference.tokens_to_generate=256 \
quantization.algorithm=awq \
quantization.enable=False

```
example format of .jsonl prompt_file:
```
{"video": "video_test.mp4", "text": "Can you describe the scene?", "category": "conv", "question_id": 0}
```
[input video file](assets/video_test.mp4)

output:
```
<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
Can you describe the scene?<video>
<extra_id_1>Assistant
<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:4
Hand with a robot arm

CLEAN RESPONSE: Hand with a robot arm
```


### inference with finetuned video neva model (no need to specify base LM)

An example of an inference script execution:

For running video inference:

```
CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /path/to/neva_evaluation.py \
--config-path=/path/to/conf/ \
--config-name=video_neva_inference.yaml
tensor_model_parallel_size=4 \
pipeline_model_parallel_size=1 \
neva_model_file=/path/to/video/neva/model \
trainer.devices=4 \
trainer.precision=bf16 \
prompt_file=/path/to/prompt/file \
inference.videos_base_path=/path/to/videos \
inference.media_type=video \
output_file=/path/for/output/file/ \
inference.temperature=0.2 \
inference.top_k=0 \
inference.top_p=0.9 \
inference.greedy=False \
inference.add_BOS=False \
inference.all_probs=False \
inference.repetition_penalty=1.2 \
inference.insert_media_token=right \
inference.tokens_to_generate=256 \
quantization.algorithm=awq \
quantization.enable=False

```

