# End-to-End Spoken Language Intent Classification and Slot Filling on SLURP Dataset

## Introduction
This example shows how to train an end-to-end model for spoken language understanding on the SLURP dataset [2]. The model is an encoder-decoder framework, where the encoder is a Conformer-large [3] model initialized from [here](https://ngc.nvidia.com/models/nvidia:nemo:stt_en_conformer_ctc_large), while the decoder is a Transformer decoder [4] randomly initialized. The model is trained by minimizing the negative log-likelihood loss with teacher forcing and label smoothing.

## Results

We present the main results of our models, as well as that of some baselines, in the following table.
|                                                  |                |                          | **Intent (Scenario_Action)** |               | **Entity** |        |              | **SLURP Metrics** |                     |
|--------------------------------------------------|----------------|--------------------------|------------------------------|---------------|------------|--------|--------------|-------------------|---------------------|
|                     **Model**                    | **Params (M)** |      **Pretrained**      |         **Accuracy**         | **Precision** | **Recall** | **F1** | **Precsion** |     **Recall**    |        **F1**       |
| NeMo-Conformer-Transformer-Large                 | 127            | NeMo ASR-Set 3.0         |                        90.14 |         86.46 |      82.29 |  84.33 |        84.31 |             80.33 |               82.27 |
| NeMo-Conformer-Transformer-Large                 | 127            | NeMo SSL-LL60kh          |                        89.04 |         73.19 |       71.8 |  72.49 |         77.9 |             76.65 |               77.22 |
| NeMo-Conformer-Transformer-Large                 | 127            | None                     |                        72.56 |         43.19 |       43.5 |  43.34 |        53.59 |             53.92 |               53.76 |
| NeMo-Conformer-Transformer-XLarge                | 617            | NeMo SSL-LL60kh          |                        91.04 |         76.67 |      74.36 |  75.49 |        82.44 |             80.14 |               81.28 |

Note: LL60kh refers to the Libri-Light dataset [7].  

## Usage
Please install [NeMo](https://github.com/NVIDIA/NeMo) [1] before proceeding. 

### Install Dependencies
Under the current directory, run
```bash
pip install -r requirements.txt
```

### Data Preparation
1. Under the current directory, run the following script to download the audio data, annotaion files and evaluation code.
```bash
DATA_DIR="./slurp_data"
mkdir -p $DATA_DIR

echo "Downloading slurp audio data..."
wget https://zenodo.org/record/4274930/files/slurp_real.tar.gz -P $DATA_DIR
wget https://zenodo.org/record/4274930/files/slurp_synth.tar.gz -P $DATA_DIR

echo "Extracting audio files to ${DATA_DIR}/slurp*"
tar -zxvf $DATA_DIR/slurp_real.tar.gz -C $DATA_DIR
tar -zxvf $DATA_DIR/slurp_synth.tar.gz -C $DATA_DIR

echo "Downloading annotations..."
mkdir -p $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/test.jsonl -P $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/devel.jsonl -P $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/train_synthetic.jsonl -P $DATA_DIR/raw_annotations
wget https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/train.jsonl -P $DATA_DIR/raw_annotations

echo "Downloading evaluation code..."
wget https://github.com/pswietojanski/slurp/raw/master/scripts/evaluation/util.py -P eval_utils/evaluation
wget https://github.com/pswietojanski/slurp/raw/master/scripts/evaluation/metrics/distance.py -P eval_utils/evaluation/metrics
wget https://github.com/pswietojanski/slurp/raw/master/scripts/evaluation/metrics/metrics.py -P eval_utils/evaluation/metrics

echo "Done."
```

2. Prepare the manifests by running: 
```bash
DATA_DIR="./slurp_data"
RAW_ANNO_DIR="${DATA_DIR}/raw_annotations"
MANIFESTS_DIR="${DATA_DIR}/raw_manifests"

echo "Preparing manifests..."
python data_utils/prepare_slurp.py --data_root $RAW_ANNO_DIR --output $MANIFESTS_DIR

echo "Decoding audios and updating manifests..."
python data_utils/decode_resample.py --data_root $DATA_DIR --manifest $MANIFESTS_DIR
```

### Building Tokenizers
1. Build the tokenizer for slu by running:
```bash
DATA_ROOT="./slurp_data"
python ../../../scripts/tokenizers/process_asr_text_tokenizer.py \
  --manifest="${DATA_ROOT}/train_slu.json,${DATA_ROOT}/train_synthetic_slu.json" \
  --data_root="${DATA_ROOT}/tokenizers_slu/" \
  --vocab_size=58 \
  --tokenizer="spe" \
  --spe_type="unigram" \
  --log \
  --spe_bos \
  --spe_eos \
  --spe_pad
```


### Training
Run with the default config that uses ASR-pretrained encoder on NeMo ASR-set 3.0. The default batch size is set to 16 for a GPU with 32GB memory, please adjust it to your own case. Training for 100 epochs takes around 18 hours on a single RTX A6000 GPU with 49GB memory.

```bash
DATA_DIR="./slurp_data"
CUDA_VISIBLE_DEVICES=0 python run_speech_intent_slot_train.py \
    --config-path="./configs" --config-name=conformer_transformer_large_bpe \
    model.train_ds.manifest_filepath="[${DATA_DIR}/train_slu.json,${DATA_DIR}/train_synthetic_slu.json]" \
    model.validation_ds.manifest_filepath="${DATA_DIR}/devel_slu.json" \
    model.test_ds.manifest_filepath="${DATA_DIR}/test_slu.json" \
    model.tokenizer.dir="${DATA_DIR}/tokenizers_slu/tokenizer_spe_unigram_v58_pad_bos_eos" \
    model.train_ds.batch_size=16 \
    model.validation_ds.batch_size=16 \
    model.test_ds.batch_size=16 \
    trainer.devices=1 \
    trainer.max_epochs=100 \
    model.optim.sched.warmup_steps=2000 \
    exp_manager.create_wandb_logger=false
```


### Evaluation
After trainng, we can evaluate the model by running the following script, which will first perform checkpoint averaging and then run beam search with the averaged checkpoint on the test set.
```bash
DATA_DIR="./slurp_data"
EXP_NAME="slurp_conformer_transformer_large"
CKPT_DIR="./nemo_experiments/${EXP_NAME}/checkpoints"

python ../../../scripts/checkpoint_averaging/checkpoint_averaging.py ${CKPT_DIR}

NEMO_MODEL="${CKPT_DIR}/${EXP_NAME}-averaged.nemo"
CUDA_VISIBLE_DEVICES=0 python run_speech_intent_slot_eval.py \
    dataset_manifest="${DATA_DIR}/test_slu.json" \
    model_path=${NEMO_MODEL} \
    batch_size=32 \
    num_workers=8 \
    searcher.type="beam" \
    searcher.beam_size=32 \
    searcher.temperature=1.25 \
    only_score_manifest=false
```

### Using Encoder Finetuned on SLURP Speech Recognition
To learn how to finetune the Conformer encoder on SLURP ASR, please refer to the tutorials at 
- [Finetuning CTC models on other languages](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb)
- [Self-Supervised pre-training for ASR](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Self_Supervised_Pre_Training.ipynb)


## Pretrained Models
The pretrained models and directions on how to use them are available [here](https://ngc.nvidia.com/catalog/models/nvidia:nemo:slu_conformer_transformer_large_slurp).


## Reference
[1] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

[2] [SLURP: A Spoken Language Understanding Resource Package](https://arxiv.org/abs/2011.13205)

[3] [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)

[4] [Attention Is All You Need](https://arxiv.org/abs/1706.03762?context=cs)

[5] [Integration of Pre-trained Networks with Continuous Token Interface for End-to-End Spoken Language Understanding](https://arxiv.org/abs/2104.07253)

[6] [SpeechBrain SLURP Recipe](https://github.com/speechbrain/speechbrain/tree/develop/recipes/SLURP)

[7] [Libri-Light: A Benchmark for ASR with Limited or No Supervision](https://arxiv.org/abs/1912.07875)

## Acknowledgments

The evaluation code is borrowed from the official [SLURP package](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation), and some data processing code is adapted from [SpeechBrain SLURP Recipe](https://github.com/speechbrain/speechbrain/tree/develop/recipes/SLURP).