
# Environment Setup
docker run --gpus all -it --rm -v $PWD/NeMo:/opt/NeMo -v $PWD:/ws --shm-size=128g -p 8888:8888 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:dev

## convert tokenizer for finetuning or inference
We need to first add the time tokens to the llama/llava tokenizer.
First download the tokenizer [here](https://huggingface.co/liuhaotian/llava-v1.5-13b/blob/main/tokenizer.model).

### LITA 1.0 tokenizer
For LITA 1.0, we need to add 100 time tokens to the base tokenizer (llava/llama) according to the [LITA paper](https://arxiv.org/pdf/2403.19046).

We can do this by:
```
cd /opt && git clone https://github.com/google/sentencepiece.git
cd /opt/sentencepiece/src/
protoc --python_out=/opt/NeMo/scripts/tokenizers/ sentencepiece_model.proto
input_hf_tokenizer=/ws/<tokenizer just downloaded>
output_nemo_tokenizer=/ws/converted_nemo_model/tokenizer_1_0.model

python /opt/NeMo/scripts/tokenizers/add_special_tokens_to_sentencepiece.py \
        --input_file $input_hf_tokenizer \
        --output_file $output_nemo_tokenizer \
        --is_userdefined \
        --tokens "<t0>" "<t1>" "<t2>" "<t3>" "<t4>" "<t5>" "<t6>" "<t7>" "<t8>" "<t9>" "<t10>" "<t11>" "<t12>" "<t13>" "<t14>" "<t15>" "<t16>" "<t17>" "<t18>" "<t19>" "<t20>" "<t21>" "<t22>" "<t23>" "<t24>" "<t25>" "<t26>" "<t27>" "<t28>" "<t29>" "<t30>" "<t31>" "<t32>" "<t33>" "<t34>" "<t35>" "<t36>" "<t37>" "<t38>" "<t39>" "<t40>" "<t41>" "<t42>" "<t43>" "<t44>" "<t45>" "<t46>" "<t47>" "<t48>" "<t49>" "<t50>" "<t51>" "<t52>" "<t53>" "<t54>" "<t55>" "<t56>" "<t57>" "<t58>" "<t59>" "<t60>" "<t61>" "<t62>" "<t63>" "<t64>" "<t65>" "<t66>" "<t67>" "<t68>" "<t69>" "<t70>" "<t71>" "<t72>" "<t73>" "<t74>" "<t75>" "<t76>" "<t77>" "<t78>" "<t79>" "<t80>" "<t81>" "<t82>" "<t83>" "<t84>" "<t85>" "<t86>" "<t87>" "<t88>" "<t89>" "<t90>" "<t91>" "<t92>" "<t93>" "<t94>" "<t95>" "<t96>" "<t97>" "<t98>" "<t99>" "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" "<extra_id_4>" "<extra_id_5>" "<extra_id_6>" "<extra_id_7>" "<extra_id_8>" "<extra_id_9>"
```
You may wonder what these `extra_id_x` tokens mean. These are special tokens reserved in NeMo. Check [this file](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/multimodal/data/neva/conversation.py) for more details.

Here are some extra tokens' definition:
```
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_SYSTEM_TOKEN = "<extra_id_0>"
DEFAULT_SEPARATOR_TOKEN = "<extra_id_1>"
DEFAULT_LABELS_TOKEN = "<extra_id_2>"
DEFAULT_IMAGE_PATCH_TOKEN = "<extra_id_3>"
DEFAULT_IM_START_TOKEN = "<extra_id_4>"
DEFAULT_IM_END_TOKEN = "<extra_id_5>"
DEFAULT_BOS_TOKEN = "<extra_id_6>"
DEFAULT_EOS_TOKEN = "<extra_id_7>"
DEFAULT_VID_START_TOKEN = "<extra_id_8>"
DEFAULT_VID_END_TOKEN = "<extra_id_9>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
```

### LITA 1.5 tokenizer
In LITA 1.5, we added `vid_start`, `vid_end`, `img_start` and `img_end` to separate temporal tokens and spatial tokens. The finetuned LITA1.5 checkpoints put these special tokens before the time tokens. Therefore, we can convert the tokenizer simply by:
```
input_hf_tokenizer=pretrained_models/<tokenizer just downloaded>
output_nemo_tokenizer=converted_nemo_model/tokenizer_1_5.model
python /opt/NeMo/scripts/tokenizers/add_special_tokens_to_sentencepiece.py \
        --input_file $input_hf_tokenizer \
        --output_file $output_nemo_tokenizer \
        --is_userdefined \
        --tokens "<extra_id_4>" "<extra_id_5>" "<extra_id_8>" "<extra_id_9>" "<t0>" "<t1>" "<t2>" "<t3>" "<t4>" "<t5>" "<t6>" "<t7>" "<t8>" "<t9>" "<t10>" "<t11>" "<t12>" "<t13>" "<t14>" "<t15>" "<t16>" "<t17>" "<t18>" "<t19>" "<t20>" "<t21>" "<t22>" "<t23>" "<t24>" "<t25>" "<t26>" "<t27>" "<t28>" "<t29>" "<t30>" "<t31>" "<t32>" "<t33>" "<t34>" "<t35>" "<t36>" "<t37>" "<t38>" "<t39>" "<t40>" "<t41>" "<t42>" "<t43>" "<t44>" "<t45>" "<t46>" "<t47>" "<t48>" "<t49>" "<t50>" "<t51>" "<t52>" "<t53>" "<t54>" "<t55>" "<t56>" "<t57>" "<t58>" "<t59>" "<t60>" "<t61>" "<t62>" "<t63>" "<t64>" "<t65>" "<t66>" "<t67>" "<t68>" "<t69>" "<t70>" "<t71>" "<t72>" "<t73>" "<t74>" "<t75>" "<t76>" "<t77>" "<t78>" "<t79>" "<t80>" "<t81>" "<t82>" "<t83>" "<t84>" "<t85>" "<t86>" "<t87>" "<t88>" "<t89>" "<t90>" "<t91>" "<t92>" "<t93>" "<t94>" "<t95>" "<t96>" "<t97>" "<t98>" "<t99>" "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" "<extra_id_6>" "<extra_id_7>"
```

If you plan to start finetuning from a llava model, it should be fine whether you use lita1.0 or lita1.5 tokenizer. No need to worry about the order of these tokens.

You can load the tokenizer and check it by:
```
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
tokenizer_path='XXXX/XXX/TOKENIZER.model'
tokenizer = get_nmt_tokenizer(library="sentencepiece", tokenizer_model=tokenizer_path)
```

## Convert finetuned checkpoints
Before converting the checkpoint, we should download the base model llava by:
```
cd /opt/ && git clone https://github.com/haotian-liu/LLaVA   # we only need the model structure, no need to install
export PYTHONPATH=/opt/LLaVA/:$PYTHONPATH
cd /ws  # do not run the below commands under `/opt` folder
```

In this part, we'll show how to convert the [LITA1.0 checkpoints](https://github.com/NVlabs/LITA/tree/main?tab=readme-ov-file#weights)

First download checkpoints from [here](https://drive.google.com/drive/u/0/folders/1-P7p-tq5aXZzSoefEJx4PSFKH8jt8KWy)

Convert pretrained lita model to nemo model
1. download lita1.0 weights from [here](https://drive.google.com/drive/u/0/folders/1-P7p-tq5aXZzSoefEJx4PSFKH8jt8KWy)
2. Run
```
input_hf_ckpt=/ws/lita-vicuna-v1-3-13b-finetune/
output_nemo_model=/ws/converted_nemo_model/lita-vicuna-v1-3-13b-finetune.nemo
nemo_tokenizer=/ws/converted_nemo_model/tokenizer_1_0.model  # use the lita1.0 nemo tokenizer file path you just saved
config_file=lita_config.yaml  # check the config file in /NeMo/examples/multimodal/multimodal_llm/neva/conf/lita_config.yaml
python /opt/NeMo/examples/multimodal/multimodal_llm/neva/convert_hf_llava_to_neva.py \
--in-file $input_hf_ckpt \
--out-file $output_nemo_model \
--tokenizer-model $nemo_tokenizer \
--config-file $config_file \
--conv-template v1
```

For internal users or developers, suppose your lita1.5 model is `llava-v1.5-13b-lita-im-se-didemo-charades-e024`. You can run the below commands to convert:

```
input_hf_ckpt=/ws/finetuned_models/llava-v1.5-13b-lita-im-se-didemo-charades-e024
nemo_tokenizer=/ws/converted_nemo_model/tokenizer_1_5.model
config_file=lita_1_5_config.yaml
output_nemo_model=/ws/converted_nemo_model/llava-v1.5-13b-lita-im-se-didemo-charades-e024.nemo
python /opt/NeMo/examples/multimodal/multimodal_llm/neva/convert_hf_llava_to_neva.py \
    --in-file $input_hf_ckpt \
    --out-file $output_nemo_model \
    --tokenizer-model $nemo_tokenizer \
    --config-file $config_file \
    --conv-template v1
```

Always remember to check these options in `lita_x_config.yaml`, `lita_video_arch`, `visual_token_format`, `sample_frames` (lita 1.5, `num_frames`), `num_frames` (lita 1.5, `max_vid_se_frames`).


## convert pretrained checkpoints for finetuning

download the pretrained weights from huggingface:
```
# Option 1 (preferred)
git clone https://huggingface.co/Lin-Chen/ShareGPT4V-13B_Pretrained_vit-large336-l12  # vision tower
# pre-trained LLM and projector
git clone https://huggingface.co/Lin-Chen/ShareGPT4V-13B_Pretrained_vit-large336-l12_vicuna-13b-v1.5

# Option 2
# LLaVA-1.5
git clone https://huggingface.co/lmsys/vicuna-13b-v1.5   # pretrained llm 

git clone https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5  # pretrain_mm_mlp_adapter
```

To convert option1 to LITA1.5
```
input_hf_ckpt=/ws/pretrained_models/ShareGPT4V-13B_Pretrained_vit-large336-l12_vicuna-13b-v1.5
nemo_tokenizer=/ws/converted_nemo_model/tokenizer_1_5.model
config_file=lita_1_5_config.yaml
output_nemo_model=/ws/converted_nemo_model/ShareGPT4V-13B_Pretrained_vit-large336-l12_vicuna-13b-v1.5.nemo
python /opt/NeMo/examples/multimodal/multimodal_llm/neva/convert_hf_llava_to_neva.py \
    --in-file $input_hf_ckpt \
    --out-file $output_nemo_model \
    --tokenizer-model $nemo_tokenizer \
    --config-file $config_file \
    --conv-template v1  # v1-> vicuna, nvgpt, llama_2
```

To convert option2 to LITA1.5
```
input_hf_ckpt=/ws/pretrained_models/vicuna-13b-v1.5
mm_projector_ckpt_dir=/ws/pretrained_models/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5
nemo_tokenizer=/ws/converted_nemo_model/tokenizer_1_5.model
config_file=lita_1_5_config.yaml
output_nemo_model=/ws/converted_nemo_model/vicuna-13b-v1.5-mm-projector.nemo

python /opt/NeMo/examples/multimodal/multimodal_llm/neva/convert_hf_llava_to_neva.py \
    --in-file $input_hf_ckpt \
    --out-file $output_nemo_model \
    --tokenizer-model $nemo_tokenizer \
    --config-file $config_file \
    --conv-template v1 \
    --mm_projector_ckpt_dir $mm_projector_ckpt_dir
```

Notice the `convert_hf_llava_to_neva.py` script will automatically expand the vocab size and convert the embedding size to be divisible by some base value.


## Finetuning with nemo pretrained model

### Dataset Preprocessing
The dataset file format for finetuning should be like:
```
[
    # 1st example: video question answer
    {
        "id": "1043215450",
        "video": "076101_076150/1043215450.mp4",   # video_path will be prepended
        "conversations": 
        [
            {"from": "human", "value": "<video>\n is the athlete wearing trousers"}, 
            {"from": "gpt", "value": "Yes"}
        ]       
    },
    # 2nd example: dense video captioning
    {
        "id": "xxxx",
        "video: "xxxx.mp4",
        "conversations":
        [
            {"from": "human", "value": "<video>\n "Provide a detailed description of the given video.Prepend each sentence with its start and end timestamps."}, 
            {"from": "gpt", "value": "<t1> <t2> Apply eyeshadow on the crease with brush <t3> <t4> Apply eyeshadow on the outer corner of eyes with brush"}
        ]
    },
    # 3rd example: event classification
    {
        "id": "xxxx",
        "video: "xxxx.mp4",
        "conversations":
        [
            {"from": "human", "value": "<video>\n "What is the action performed in this video?"}, 
            {"from": "gpt", "value": "brush hair"}
        ]

    },
    # 4th example: event localization
    {
        "id": "-4RXOT_UfpM_2",
        "video": "-4RXOT_UfpM_2.mp4",
        "conversations": [
            {"from": "human", "value": "<video>\nWhen is \"Apply concealer on the eyelids and blend with sponge\" depicted in the video? Provide a response using only start and end timestamps."},
            {"from": "gpt", "value": "<t4> <t18>"}
        ],
        "durations": 119.01901901901903
    },
    ...
]
```

Here the `<video>` is the placeholder for the video features. In the 2nd example, `<t1>` `<t2>` are the time tokens to indidate in which time interval we've seen this event or description of the time inverval. You can prepare your time tokens like this:
```python
import numpy as np
TIME_TOKEN_TEMPLATE = "<t{t}>"
def time_to_string(time, num_time_tokens):
    max_offset = float(num_time_tokens - 1)
    time = int(np.round(max_offset * time))
    return TIME_TOKEN_TEMPLATE.format(t=time)

# example of converting time tokens
# from 10seconds to 15 seconds
num_time_tokens = 100
start = 10.0   # the 10 seconds
end = 15.0     # the 15 seconds
duration = 200.0 # total video duration is 200seconds
start = start / duration 
end = end / duration
start_time_token_str = time_to_string(start, num_time_tokens)
end_time_token_str = time_to_string(end, num_time_tokens,)
```

We also provide scripts to help convert the dense video caption dataset to nemo training dataset.
Please refer to  `convert_dvc_dataset_for_training.py` and `convert_dvc_dataset_for_evaluation.py` under `NeMo/scripts/multimodal_dataset_conversion/` for how to convert the dataset for training and evaluation.

If you want to augment your dvc dataset by using external LLM API, you may refer to `generate_qa_data.py` and `convert_video_qa_dataset.py` under the same directory.

### Finetuning

Below is an example command to do finetuning. (A100x8 GPUs)
```
ANDB_NAME=nemo_lita_finetuning
WANDB_PROJECT=nemo_lita
EXP_MANAGER_DIR=/ws/train
video_folder=/ws/dataset/videos
data_path=/ws/dataset/train.json
# model path is the llava-similar nemo model or a converted nemo model constructed by the above steps
model_path=/ws/converted_nemo_model/llava-v1.5-7b-lita.nemo
pretrained_hf_vision_model=/ws/pretrained_models/ShareGPT4V-13B_Pretrained_vit-large336-l12
num_gpus=8
wandb login <YOUR WANDB API>
torchrun --nproc_per_node=${num_gpus} /opt/NeMo/examples/multimodal/multimodal_llm/neva/neva_finetune.py \
  --config-path=/opt/NeMo/examples/multimodal/multimodal_llm/neva/conf/ \
  --config-name=lita_1_5_config.yaml \
  ++cluster_type=BCP \
  exp_manager.wandb_logger_kwargs.name=${WANDB_NAME} \
  exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT} \
  exp_manager.exp_dir=${EXP_MANAGER_DIR} \
  trainer.num_nodes=1 \
  trainer.precision=bf16 \
  trainer.devices=${num_gpus} \
  trainer.max_steps=262 \
  trainer.val_check_interval=100 \
  trainer.limit_val_batches=5 \
  model.megatron_amp_O2=false \
  model.mm_cfg.llm.freeze=false \
  model.mm_cfg.vision_encoder.freeze=true \
  model.mm_cfg.vision_encoder.from_pretrained=$pretrained_hf_vision_model \
  model.global_batch_size=128 \
  model.micro_batch_size=1 \
  model.tensor_model_parallel_size=4 \
  model.pipeline_model_parallel_size=1 \
  model.restore_from_path=${model_path} \
  model.context_parallel_size=1 \
  model.data.video_folder=${video_folder} \
  model.data.data_path=${data_path} \
  model.mm_cfg.use_lita=true \
  model.mm_cfg.lita.lita_video_arch=temporal_all_resolution \
  model.mm_cfg.lita.visual_token_format=im_vid_start_end \
  model.mm_cfg.lita.sample_frames=4 \
  model.mcore_gpt=true \
  model.transformer_engine=true \
  model.optim.sched.warmup_steps=15
```


## Inference with finetuned weights

Run inference:
```
neva_model_file=/ws/converted_nemo_model/llava-v1.5-13b-lita-finetuned.nemo
prompt_file=/ws/test/prompt_file.json
output_file=/ws/test/output.json
video_base_path=/ws/test/videos
num_gpus=1
torchrun --nproc_per_node=$num_gpus /ws/NeMo/examples/multimodal/multimodal_llm/neva/neva_evaluation.py \
    --config-path=/opt/NeMo/examples/multimodal/multimodal_llm/neva/conf/ \
    --config-name=neva_inference.yaml \
    tensor_model_parallel_size=$num_gpus \
    pipeline_model_parallel_size=1 \
    neva_model_file=$neva_model_file \
    trainer.devices=$num_gpus \
    trainer.precision=16 \
    prompt_file=$prompt_file \
    inference.media_base_path=$video_base_path \
    inference.media_type=video \
    output_file=$output_file \
    inference.temperature=0.2 \
    inference.top_k=0 \
    inference.top_p=0.9 \
    inference.greedy=False \
    inference.add_BOS=False \
    inference.all_probs=False \
    inference.repetition_penalty=1.2 \
    inference.insert_media_token=left \
    inference.tokens_to_generate=256 \
    quantization.algorithm=awq \
    quantization.enable=False
```

The prompt file can be one json string one line:
```
{"video": "1066647457.mp4", "text": "Can you describe the scene?", "category": "conv", "question_id": 0}
{"video": "1066647457.mp4", "text": "What's the color of the man's hoodie?", "category": "conv", "question_id": 1}
{"video": "1066647457.mp4", "text": "When does the man put his hands on the rock wall", "duration": 13.0, "category": "conv", "question_id": 1}
```

The prompt file can also be json list file:
```
[
    {
        "video": "1SX-19LDHGY_10.mp4",
        "question": "\nIs there any activity related to lip makeup application between <10s> and <20s> in the video?",
        "ref_answer": "No, there is no activity related to lip makeup application between <10s> and <20s> in the video.",
        "type": "4",
        "question_id": "v_1SX-19LDHGY_10_9",
        "duration": 42.0
    },
    {
        "vid": "3RIeQTScqEI_2",
        "question_id": "3RIeQTScqEI_2_0",
        "question": "At what time in the video does \"Apply eyeliner gel on lashline with brush\" take place? Answer the question only using start and end timestamps.",
        "duration": 103.004,
        "ref_answer": "<2> <59> Apply eyeliner gel on lashline with brush",
        "video": "3RIeQTScqEI_2.mp4"
    }
    ...
]
```

The `neva_evaluation.py` script would append `pred_answer` field to the input prompt file as the output file. You'll need to ensure `video`, `question` or `text` field is in the input prompt file. The `duration` field is required if you want to test the localization and time question. 

## Evaluation

Once you get the inference result or the `pred_answer`. You can do your evaluation easily. There are two example scripts `eval_qa.py` and `eval_rtl.py` under `NeMo/examples/multimodal/multimodal_llm/neva/eval` provided to help you with the RTL task and VQA task. They are all nemo independent, which means you could do the evaluation without using nemo.

`eval_qa.py` is a simple example to request NVIDIA LLM APIs to do evaluation on the question answer task. It's pretty straightforward.  The default one uses `llama-70b-instruct` as an example. You can explore more options [here](https://build.nvidia.com/explore/discover).

`eval_rtl.py` is used to meassure the IOU or the overlap of the predicted time interval and the reference time interval. It will also give the `precision@0.5` score. When the overlap ratio is larger than `0.5`, it will be marked as correct.


