# DCT Tokenizer

## How to run.

This guide is optimised for Qwen. I revised 'nemo/collections/vlm/qwen2vl/data/task_encoder.py' for compatibility.

1. Start the environment (Docker)

```bash
docker ./make-and-run-docker.sh
```

2. Pin Transformers to v4.51.3

```bash
pip install "transformers==4.51.3"
```

3. Download Qwen checkpoints (cached under ~/.cache/nemo/...)

```bash
python down_hf_qwen_ckpt.py
```


4. Prepare the Cambrian737k dataset. Please change your path for json file.

```bash
python data_preparation.py
```

Then run the following code

```bash
energon prepare /datasets/wds
```

Choose '11. Crude Sample' for run.

5. run
```bash
bash qwen_launch.sh
```
