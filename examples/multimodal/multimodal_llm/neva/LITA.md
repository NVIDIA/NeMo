
docker run --gpus all -it --rm -v $PWD/NeMo:/opt/NeMo -v $PWD:/ws --shm-size=8g \
     -p 8888:8888 --ulimit memlock=-1 --ulimit \
      stack=67108864 nvcr.io/nvidia/nemo:dev.framework

# nvcr.io/nvidia/nemo:24.03.01.framework   cuda_driver mismatch
# 

# convert pretrained checkpoints

download the pretrained weights from huggingface
```
# LLaVA-1.5
git clone https://huggingface.co/lmsys/vicuna-13b-v1.5   # pretrained llm 

git clone https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5  # pretrain_mm_mlp_adapter


git clone https://huggingface.co/Lin-Chen/ShareGPT4V-13B_Pretrained_vit-large336-l12  # vision tower
# pre-trained LLM and projector
git clone https://huggingface.co/Lin-Chen/ShareGPT4V-13B_Pretrained_vit-large336-l12_vicuna-13b-v1.5

```


## convert tokenizer for finetuning or inference
Please check [here](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/multimodal/data/neva/conversation.py) for the special tokens meaning.
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
DEFAULT_VID_START_TOKEN = "<extra_id_9>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
```


```
cd /opt/sentencepiece/src/
protoc --python_out=/opt/NeMo/scripts/tokenizers/ sentencepiece_model.proto

# the input hf tokenizer can be downloaded from llava hf repo as well: https://huggingface.co/liuhaotian/llava-v1.5-13b/tree/main

input_hf_tokenizer=pretrained_models/ShareGPT4V-13B_Pretrained_vit-large336-l12_vicuna-13b-v1.5/tokenizer.model
output_hf_tokenizer=converted_nemo_model/tokenizer.model

! python /opt/NeMo/scripts/tokenizers/add_special_tokens_to_sentencepiece.py \
        --input_file $input_hf_tokenizer \
        --output_file $output_hf_tokenizer \
        --is_userdefined \
        --tokens "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" \
             "<extra_id_4>" "<extra_id_5>" "<extra_id_6>" "<extra_id_7>" \
            "<extra_id_8>" "<extra_id_9>"
```

To finetune or do inference LITA 1.0
*100 time tokens*  add to the original llava tokenizer
tokens "<t0>" "<t1>" "<t2>" "<t3>" "<t4>" "<t5>" "<t6>" "<t7>" "<t8>" "<t9>" "<t10>" "<t11>" "<t12>" "<t13>" "<t14>" "<t15>" "<t16>" "<t17>" "<t18>" "<t19>" "<t20>" "<t21>" "<t22>" "<t23>" "<t24>" "<t25>" "<t26>" "<t27>" "<t28>" "<t29>" "<t30>" "<t31>" "<t32>" "<t33>" "<t34>" "<t35>" "<t36>" "<t37>" "<t38>" "<t39>" "<t40>" "<t41>" "<t42>" "<t43>" "<t44>" "<t45>" "<t46>" "<t47>" "<t48>" "<t49>" "<t50>" "<t51>" "<t52>" "<t53>" "<t54>" "<t55>" "<t56>" "<t57>" "<t58>" "<t59>" "<t60>" "<t61>" "<t62>" "<t63>" "<t64>" "<t65>" "<t66>" "<t67>" "<t68>" "<t69>" "<t70>" "<t71>" "<t72>" "<t73>" "<t74>" "<t75>" "<t76>" "<t77>" "<t78>" "<t79>" "<t80>" "<t81>" "<t82>" "<t83>" "<t84>" "<t85>" "<t86>" "<t87>" "<t88>" "<t89>" "<t90>" "<t91>" "<t92>" "<t93>" "<t94>" "<t95>" "<t96>" "<t97>" "<t98>" "<t99>" "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" "<extra_id_4>" "<extra_id_5>" "<extra_id_6>" "<extra_id_7>" "<extra_id_8>" "<extra_id_9>"

To finetune or do inference LITA 1.5
*img_start_end True* *vid_start_end True*  add to the pretrained model and 100 time tokens.
token "<extra_id_4>" "<extra_id_5>" "<extra_id_8>" "<extra_id_9>" "<t0>" "<t1>" "<t2>" "<t3>" "<t4>" "<t5>" "<t6>" "<t7>" "<t8>" "<t9>" "<t10>" "<t11>" "<t12>" "<t13>" "<t14>" "<t15>" "<t16>" "<t17>" "<t18>" "<t19>" "<t20>" "<t21>" "<t22>" "<t23>" "<t24>" "<t25>" "<t26>" "<t27>" "<t28>" "<t29>" "<t30>" "<t31>" "<t32>" "<t33>" "<t34>" "<t35>" "<t36>" "<t37>" "<t38>" "<t39>" "<t40>" "<t41>" "<t42>" "<t43>" "<t44>" "<t45>" "<t46>" "<t47>" "<t48>" "<t49>" "<t50>" "<t51>" "<t52>" "<t53>" "<t54>" "<t55>" "<t56>" "<t57>" "<t58>" "<t59>" "<t60>" "<t61>" "<t62>" "<t63>" "<t64>" "<t65>" "<t66>" "<t67>" "<t68>" "<t69>" "<t70>" "<t71>" "<t72>" "<t73>" "<t74>" "<t75>" "<t76>" "<t77>" "<t78>" "<t79>" "<t80>" "<t81>" "<t82>" "<t83>" "<t84>" "<t85>" "<t86>" "<t87>" "<t88>" "<t89>" "<t90>" "<t91>" "<t92>" "<t93>" "<t94>" "<t95>" "<t96>" "<t97>" "<t98>" "<t99>" "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" "<extra_id_6>" "<extra_id_7>"

The time tokens are necessary but the `extra_id_x` are special tokens in case you want to add tokens such as image start, image end tokens and system prompt tokens. You can remove them if you are sure you will not use some of them in the conv template but it's not suggested.


待考证:
To convert LITA 1.5 Finetuned model for inference
if the LITA1.5 finetuned model tokenizer has `<img_start>`, `<img_end>`, `<vid_start>`, `<vid_end token>`, you don't need to add them but you need to change its token name manually to `<extra_id_4>` `<extra_id_5>` `<extra_id_8>` `<extra_id_9>` respectively by editing the field `additional_special_tokens` and `added_tokens_decoder` in  `tokenizer_config.json`.
Then add the left tokens "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" "<extra_id_6>" "<extra_id_7>".



check tokenizer
```
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
tokenizer_path='XXXX/XXX/TOKENIZER.model'
tokenizer = get_nmt_tokenizer(library="sentencepiece", tokenizer_model=tokenizer_path)
```

Before converting the checkpoint, we should download llava by:
```
git clone https://github.com/haotian-liu/LLaVA   # we only need the model structure, no need to install
export PYTHONPATH=<the path to>/LLaVA/
``` 

## convert checkpoint to nemo model
```
input_hf_ckpt=/ws/pretrained_models/ShareGPT4V-13B_Pretrained_vit-large336-l12_vicuna-13b-v1.5/
output_nemo_model=/ws/converted_nemo_model/ShareGPT4V-13B_Pretrained_vit-large336-l12_vicuna-13b-v1.5.nemo
tokenizer_path=/ws/converted_nemo_model/tokenizer_lita15.model  # use the tokenizer file path you just saved
config_file=video_neva_config.yaml
python /opt/NeMo/examples/multimodal/multimodal_llm/neva/convert_hf_llava_to_neva.py \
--in-file $input_hf_ckpt \
--out-file $output_nemo_model \
--tokenizer-model $tokenizer_path \
--config-file $config_file
```

Notice the `convert_hf_llava_to_neva.py` script will automatically convert the vocab size from `32110` to `32128` and convert the embedding size from `32000` to `32128`.  (divisible 有关)

Convert pretrained lita model to nemo model
1. download lita1.0 weights from [here](https://drive.google.com/drive/u/0/folders/1-P7p-tq5aXZzSoefEJx4PSFKH8jt8KWy)
2. Run
```
input_hf_ckpt=/ws/lita-vicuna-v1-3-13b-finetune/
output_nemo_model=/ws/converted_nemo_model/lita-vicuna-v1-3-13b-finetune.nemo
tokenizer_path=/ws/converted_nemo_model/tokenizer.model  # use the lita1.0 nemo tokenizer file path you just saved
config_file=lita_config.yaml
python /opt/NeMo/examples/multimodal/multimodal_llm/neva/convert_hf_llava_to_neva.py \
--in-file $input_hf_ckpt \
--out-file $output_nemo_model \
--tokenizer-model $tokenizer_path \
--config-file $config_file

```


Run inference:
```
neva_model_file=/ws/converted_nemo_model/lita-vicuna-v1-3-13b-finetune.nemo
prompt_file=/ws/test/prompt_file.json
output_file=/ws/test/output.json
video_base_path=/ws/test
torchrun --nproc_per_node=1 /opt/NeMo/examples/multimodal/multimodal_llm/neva/neva_evaluation.py \
    --config-path=/opt/NeMo/examples/multimodal/multimodal_llm/neva/conf/ \
    --config-name=neva_inference.yaml \
    tensor_model_parallel_size=1 \
    pipeline_model_parallel_size=1 \
    neva_model_file=$neva_model_file \
    trainer.devices=1 \
    trainer.precision=bf16 \
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
    inference.insert_media_token=right \
    inference.tokens_to_generate=256 \
    quantization.algorithm=awq \
    quantization.enable=False
```