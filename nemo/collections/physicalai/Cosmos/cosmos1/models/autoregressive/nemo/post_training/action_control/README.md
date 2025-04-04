# Action-Control Autoregressive Post-Training

1. Run the following command to download and start the container:

    ```bash
    docker run --ipc=host -it --gpus=all \
    -v $PATH_TO_COSMOS_REPO:/workspace/Cosmos \
    -v <path/to/store/checkpoints>:/root/.cache/huggingface \
    nvcr.io/nvidia/nemo:25.02.rc1 bash
    ```

2. Set the following environment variables:

    ```
    pip install --no-cache-dir imageio[ffmpeg] pyav iopath
    export HF_HOME=/root/.cache/huggingface
    export HF_TOKEN=<your/HF/access/token>
    export PYTHONPATH=/workspace/Cosmos:$PYTHONPATH
    ```

3. Run the following command to download the models:

   ```bash
   python Cosmos/cosmos1/models/autoregressive/nemo/download_autoregressive_nemo.py
   ```
> [!NOTE]
> The full bridge dataset is approximately 30Gb, and the full download can take approximately 2
> hours depending on your connection speed. To use a smaller, 10Mb sanity-sized dataset bundled with
> the Cosmos repository, pass `--dataset-dir Cosmos/cosmos1/models/autoregressive/assets/bridge` to
> the commands below.

4. Then run the `prepare_dataset` script for the train, test, and val splits. The bridge dataset
will be downloaded automatically the first time you run it, and stored in
`HF_HOME/assets/cosmos/action-control/datasets/`. By default, the tokenized videos will be stored in
`HF_HOME/assets/cosmos/action-control/autoregressive/bridge/`.

    ```
    python Cosmos/cosmos1/models/diffusion/nemo/post_training/action_control/prepare_dataset.py \
        --dataset-split train --batch-size 50 --num-workers 16

    python Cosmos/cosmos1/models/diffusion/nemo/post_training/action_control/prepare_dataset.py \
        --dataset-split val --batch-size 50 --num-workers 16

    python Cosmos/cosmos1/models/diffusion/nemo/post_training/action_control/prepare_dataset.py \
        --dataset-split test --batch-size 50 --num-workers 16
    ```

5. Run post-training. If no additional paths are provided, the pre-processed data will be loaded
   from `HF_HOME`.

    ```
    export NUM_DEVICES=8
    # Data-parallel batch size per TP group. On 8xA100, we use 16 for 5B and 4 for 13B.
    export MICRO_BATCH_SIZE=16
    export WANDB_API_KEY="</your/wandb/api/key>"
    export WANDB_PROJECT_NAME="cosmos-autoregressive-nemo-finetuning"
    export WANDB_RUN_ID="cosmos_autoregressive_5b_action_control"

    torchrun --nproc-per-node $NUM_DEVICES \
        Cosmos/cosmos1/models/autoregressive/nemo/post_training/action_control/action_control_finetuning.py \
        --log_dir ./logs \
        --max_steps 10 --save_every_n_steps 5 \
        --tensor_model_parallel_size $NUM_DEVICES \
        --micro_batch_size $MICRO_BATCH_SIZE \
        --model_path nvidia/Cosmos-1.0-Autoregressive-5B-Video2World
   ```
