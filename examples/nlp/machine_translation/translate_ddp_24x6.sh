#!/bin/bash
INSTANCE=dgx1v.32g.8.norm
PROJECT=backtranslation-de-en-wmt21
DATAID=68792
WORKSPACE=wmt_translate_models
WANDBLOGIN=1589819cfa34108320cd27634a3f764a29b211d8
set -e
ngc batch run --name "backtranslation_de_en_wmt21" --preempt RUNONCE \
    --image "nvcr.io/nvidia/pytorch:21.03-py3" \
    --ace nv-us-west-2 \
    --instance $INSTANCE \
    --commandline "export GLOO_SOCKET_IFNAME=eth0 && export NCCL_SOCKET_IFNAME=eth0 && nvidia-smi && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libsndfile1 ffmpeg && \
    pip install wandb==0.10.21 && pip install Cython && wandb login $WANDBLOGIN && \
    git clone https://github.com/NVIDIA/NeMo.git && cd NeMo && \
    pip uninstall -y torchtext && ./reinstall.sh && \
    cp -R /data/wmt21_de_en_yttm_tokens_8000 /raid/ && \
    python examples/nlp/machine_translation/translate_ddp.py \
        --model=/models/wmt21_en_ru_24x6_averaged_r2l.nemo \
        --text2translate=/raid/wmt21_de_en_yttm_tokens_8000/parallel.batches.tokens.8000._OP_0..3517_CL_.tar \
        --src_language de \
        --tgt_language en \
        --metadata_path /raid/wmt21_de_en_yttm_tokens_8000/metadata.tokens.8000.json \
        --twoside \
        --result_dir /results" \
    --result /results/ \
    --org nvidian \
    --team swdl-ai-apps \
    --datasetid $DATAID:/data/ \
    --workspace $WORKSPACE:/models/
    --datasetid $DATAID:/data/