NEMO_DIR=/home/heh/codes/nemo-main-slm
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

PROJECT_NAME=audio-text-llm-debug
EXP_NAME=AudioGPT-TSASR-LSmix2-dev-debug

# 2b checkpoint
MEGATRON_CKPT=/media/data3/speech_llm/nvgpt_ckpt/megatron_gpt_sft--validation_loss-0.488-step-1095-consumed_samples-560128.0.nemo

ALM_DIR=/media/data3/speech_llm/salm_ckpt/salm_paper_ckpt/debug0cfffF_2bsft_inf_sel_FC-GPT_2b_sft_ast_en_de_ja_ls_lr1e-4wd1e-3_CosineAnnealing_warmup2000_minlr1e-6_gbs256_mbs4_ep200
ALM_YAML=$ALM_DIR/version_0/hparams_v2.yaml
ALM_CKPT=$ALM_DIR/eval.1c_v2.ckpt

# VAL_MANIFESTS="[/home/heh/codes/nemo-main-slm/workspace/nemo_experiments/word_boosting/word_boosting_asr_manifest.json]"
# VAL_NAMES="[gtc_asr]"


TRAIN_MANIFESTS="[/home/heh/codes/nemo-main-slm/workspace/nemo_experiments/word_boosting/word_boosting_manifest_1shot.json]"
VAL_MANIFESTS="[/home/heh/codes/nemo-main-slm/workspace/nemo_experiments/word_boosting/word_boosting_manifest_1shot.json]"
VAL_NAMES="[word_boosting_1shot]"

NUM_WORKERS=0

#python \
# python -m pdb -c continue \
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python \
./run_sft_ft_audio_lm.py \
    name=$EXP_NAME \
    ++exp_manager.create_wandb_logger=false \
    ++exp_manager.name=$EXP_NAME \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.resume=false \
    trainer.devices=1 \
    model.restore_from_path=$MEGATRON_CKPT \
    model.peft.restore_from_path=$ALM_CKPT \
    model.peft.restore_from_hparams_path=$ALM_YAML \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.train_ds.num_workers=$NUM_WORKERS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    model.data.validation_ds.num_workers=$NUM_WORKERS 
