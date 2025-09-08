# DCT Tokenizer

## How to run.

This guide is optimised for Qwen. I revised 'nemo/collections/vlm/qwen2vl/data/task_encoder.py' for compatibility.

1. Start a docker environment

```bash
docker ./make-and-run-docker.sh
```

and inside docker.
```
# Pin Transformers to v4.51.3 since NeMo fw does not support Qwen2-VL under the latest version.
pip install "transformers==4.51.3"
```

2. Prepare model and datasets

```bash
# Clone the Qwen2-VL-2B-Instruct model. Expected to be donwloaded at /models/Qwen2-VL-2B-Instruct.
cd /models/
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
cd -
```

This experiments uses the [Cambrian737k dataset](https://huggingface.co/datasets/LanguageBind/Cambrian737k)
```bash
# Prepare the Cambrian737k dataset.
python data_preparation.py
energon prepare /datasets/wds
```

3. Launch training

Then run the following code to start training.
```bash
bash qwen_launch.sh
```
