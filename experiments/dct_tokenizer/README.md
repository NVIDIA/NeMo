# DCT Tokenizer Experiment

This experiment implements fine-tuning of the Qwen2-VL-2B-Instruct model using the Cambrian737k dataset with DCT (Discrete Cosine Transform) tokenization for video processing. The setup is optimized for Qwen2-VL models and includes custom modifications to the NeMo framework for compatibility.

## Setup Instructions

### 1. Environment Setup

Start the Docker environment:

```bash
bash ./make-and-run-docker.sh
```

Inside the Docker container, install the required Transformers version:

```bash
# Pin Transformers to v4.51.3 since NeMo framework does not support Qwen2-VL under the latest version
pip install "transformers==4.51.3"
```

### 2. Model Preparation

Download the Qwen2-VL-2B-Instruct model:

```bash
# Install Git LFS for large file handling
sudo apt update && sudo apt install git-lfs

# Move to models directory and clone the model
cd /models/
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
cd -
```

### 3. Dataset Preparation

This experiment uses the [Cambrian737k dataset](https://huggingface.co/datasets/LanguageBind/Cambrian737k), a large-scale multimodal dataset containing image-text conversations.

#### Download the Dataset

```bash
cd /datasets/
git clone https://huggingface.co/datasets/LanguageBind/Cambrian737k
cd -
```

**Expected dataset structure:**
```
/datasets/
└── Cambrian737k/
    ├── Cambrian737k/
    │   ├── cambrian737k.json         # Metadata file
    │   ├── ai2d.tar
    │   ├── chartqa.tar
    │   └── ... (many tar files)
    ├── lmmseval/
    └── mmvp_cache/
```

#### Convert to WebDataset Format

The raw dataset needs to be converted to WebDataset format for efficient training. The data preparation script
- Filters the dataset by checking if image files exist
- Converts images to JPEG format and stores as binary data
- Creates WebDataset shards with conversation data
- Generates metadata for efficient data loading

```bash
# Move to the experiment directory.
cd experiments/dct_tokenizer

# Convert dataset to WebDataset format
python data_preparation.py --data-dir /datasets/Cambrian737k/Cambrian737k --output-dir /datasets/Cambrian737k-wds
# Expected log messges.
# 0 conversations will be saced
# Filtering done and saved to /datasets/Cambrian737k/Cambrian737k/Cambrian737k_filtered.json.
# # writing /datasets/Cambrian737k-wds/pretrain-0.tar 0 0.0 GB 0
# 0it [00:00, ?it/s]
Dataset successfully converted to the webdataset format.
```

We now prepare dataset for Energon (NeMo's data loading system).
```bash
energon prepare /datasets/Cambrian737k-wds
# Need to select several options. Here describes how we did.

```

### 4. Launch Training

The training is configured through `qwen_launch.sh` with the following key parameters.
Launch fine-tuning process:
```bash
bash qwen_launch.sh
```

TBA

## Training Output

TBA

## Customization

### Modifying Training Parameters

Edit `qwen_launch.sh` to adjust:
- `NUM_PROC_PER_NODE`: Number of processes per node
- `DEVICES`: Number of GPUs to use
- `MBS`/`GBS`: Micro and global batch sizes
- `MINPIXELS`/`MAXPIXELS`: Image resolution range
- `EXPERIMENT_NAME`: Name for this training run


## Troubleshooting

### Common Issues

1. **Transformers Version Conflict**: Ensure you're using `transformers==4.51.3`
2. **CUDA Out of Memory**: Reduce batch size (`MBS`/`GBS`) or use fewer devices
3. **Dataset Loading Issues**: Verify the WebDataset conversion completed successfully
4. **Model Loading Errors**: Check that the model path is correct and files are downloaded

## References

- [Qwen2-VL Model](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Cambrian737k Dataset](https://huggingface.co/datasets/LanguageBind/Cambrian737k)
- [NeMo Framework](https://github.com/NVIDIA/NeMo)
- [WebDataset Documentation](https://webdataset.github.io/webdataset/)
