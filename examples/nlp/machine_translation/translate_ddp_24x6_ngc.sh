#!/bin/bash
INSTANCE=dgx1v.32g.8.norm
PROJECT=backtranslation-de-en-wmt21
DATAID=84118
WORKSPACE=wmt_translate_models
WANDBLOGIN=1589819cfa34108320cd27634a3f764a29b211d8
set -e
ngc batch run --name "translation_de_en_wmt21" --preempt RUNONCE \
    --image "nvcr.io/nvidia/pytorch:21.03-py3" \
    --ace nv-us-west-2 \
    --instance $INSTANCE \
    --commandline "export GLOO_SOCKET_IFNAME=eth0 && export NCCL_SOCKET_IFNAME=eth0 && nvidia-smi && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libsndfile1 ffmpeg && \
    pip install wandb==0.10.21 && pip install Cython && wandb login $WANDBLOGIN && \
    git clone https://github.com/sergiogcharles/NeMo.git && cd NeMo && \
    git checkout origin/nmt_distill && ./reinstall.sh && \
    cp -R /data/* /raid/ && \
    python examples/nlp/machine_translation/translate_ddp.py \
        --model=/raid/nemo_models/teacher_24_6_de_en/AAYNBase.nemo \
        --text2translate=/raid/wmt21_de_en_yttm_tokens_8000/parallel.batches.tokens.8000._OP_0..3517_CL_.tar \
        --src_language de \
        --tgt_language en \
        --metadata_path /raid/wmt21_de_en_yttm_tokens_8000/metadata.tokens.8000.json \
        --twoside \
        --result_dir /results" \
    --result /results/ \
    --org nvidian \
    --team ac-aiapps \
    --datasetid $DATAID:/data/ \
    --workspace $WORKSPACE:/models/