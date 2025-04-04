# Video2World DiT Action Control Data Pre-Processing

1. Run the following command to download and start the container:

    ```bash
    docker run --ipc=host -it --gpus=all \
    -v $PATH_TO_COSMOS_REPO:/workspace/Cosmos \
    -v $HF_HOME:/root/.cache/huggingface \
    nvcr.io/nvidian/nemo:cosmos.1.0.2 bash
    ```

2. Set the following environment variables:

    ```
    pip install --no-cache-dir imageio[ffmpeg] pyav iopath
    export HF_HOME=/root/.cache/huggingface
    export HF_TOKEN=<your/HF/access/token>
    export PYTHONPATH=/workspace/Cosmos:$PYTHONPATH
    ```

3. Run the following command to download the models. We need the continuous video tokenizer model to tokenize the action control dataset.

   ```bash
   python Cosmos/cosmos1/models/diffusion/nemo/download_diffusion_nemo.py
   ```

4. Then run the `prepare_dataset` for the train, test, and val splits. The raw bridge dataset will
be downloaded automatically the first time you run it, and stored in
`HF_HOME/assets/cosmos/action-control/datasets/`. By default, the tokenized videos will be stored in
`HF_HOME/assets/cosmos/action-control/diffusion/bridge/`, which is used to train DiT V2W. If you
already have the bridge dataset downloaded to a different directory, then you can pass
`--dataset_dir` to point to the downloaded dataset.
> [!NOTE]
> The full bridge dataset is approximately 30Gb, and the full download can take approximately 2
> hours depending on your connection speed. To use a smaller, 10Mb sanity-sized dataset bundled with
> the Cosmos repository, pass `--dataset-dir Cosmos/cosmos1/models/autoregressive/assets/bridge` to
> the commands below.

    ```
    python Cosmos/cosmos1/models/diffusion/nemo/post_training/action_control/prepare_dataset.py \
        --dataset-split train --batch-size 50 --num-workers 16

    python Cosmos/cosmos1/models/diffusion/nemo/post_training/action_control/prepare_dataset.py \
        --dataset-split val --batch-size 50 --num-workers 16

    python Cosmos/cosmos1/models/diffusion/nemo/post_training/action_control/prepare_dataset.py \
        --dataset-split test --batch-size 50 --num-workers 16
    ```
