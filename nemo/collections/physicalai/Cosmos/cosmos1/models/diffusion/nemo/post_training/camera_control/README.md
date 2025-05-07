# Video2World DiT Camera Control Data Pre-Processing

1. Run the following command to download and start the container:

    ```bash
    docker run --ipc=host -it --gpus=all \
    -v $PATH_TO_COSMOS_REPO:/workspace/Cosmos \
    -v $HF_HOME:/root/.cache/huggingface \
    nvcr.io/nvidian/nemo:cosmos.1.0.2 bash
    ```

2. Set the following environment variables:

    ```
    pip install --no-cache-dir imageio[ffmpeg] pyav iopath ffmpeg-python
    export HF_HOME=/root/.cache/huggingface
    export HF_TOKEN=<your/HF/access/token>
    export PYTHONPATH=/workspace/Cosmos:$PYTHONPATH
    ```

3. Run the following command to download the models. We need the continuous video tokenizer model to tokenize the camera control dataset.

   ```bash
   python Cosmos/cosmos1/models/diffusion/nemo/download_diffusion_nemo.py
   ```

4.  To download the DL3DV datset, we recommend that users use the `scripts/download.py` script within the [DL3DV repo](https://github.com/DL3DV-10K/Dataset) which downloads the dataset from the HuggingFace Hub.

5. Then run the `prepare_dataset.py` script to prepare the training samples for fine tuning the Cosmos-1.0-Video2World Difusion model for camera control.

    ```bash
    # Path to flattened dataset
    export RAW_DATA=<Path-to-DL3DV-Download>/DL3DV/DL3DV-ALL-4K

    # Path to Processed Dataset.
    export CACHED_DATA="./cached_camera_ctrl_data" && mkdir -p $CACHED_DATA

    # Path to intermediate processing cache
    export PROCESSING_CACHE="./video_processing_cache" && mkdir -p $PROCESSING_CACHE

    python cosmos1/models/diffusion/nemo/post_training/camera_control/prepare_dataset.py \
        --dataset_path $RAW_DATA \
        --video_processing_cache_path $PROCESSING_CACHE \
        --output_path $CACHED_DATA \
    ```

By default, the above script will construct videos from the frames provided within `DL3DV/DL3DV-ALL-4K` zip files and write them to the `--video_processing_cache_path` (under the `video_files` dir). It will then extract one 57-frame chunk from each constructed video and the video latent using the [nvidia/Cosmos-1.0-Tokenizer-CV8X8X8](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-CV8x8x8). We choose to construct the dataset in this manner to ensure that the videos align with the camera poses provided in the `transforms.json` files also included in the `DL3DV/DL3DV-ALL-4K` zip files.

6. Upon completion of this script, users should observe files of the following format writtten to their `$CACHED_DATA` directory:

```bash
0.video_latent.pth
0.t5_text_mask.pth
0.t5_text_embeddings.pth
0.plucker_embeddings.pth
0.padding_mask.pth
0.info.json
0.image_size.pth
0.conditioning_latent.pth
```
where the first integer indicates the sample index, and the trailing text provides the description of the data associated with the sample.


## Configuration Options

| Parameter                      | Description                                                                     | Default |
|--------------------------------|---------------------------------------------------------------------------------|---------|
| `--dataset_path`                   | Path to DL3DV/DL3DV-ALL-4K dataset | None |
| `--video_processing_cache_path` |  Path to where intermediate files will be stored | `./video_processing_cache` |
| `--path_to_caption_dict`  | Path to a pickled Python dictionary with keys of `"<DL3DV-sample-hash>/<chunk-num>"` and values the caption for the video chunk. If a file is not provided, a dummy caption of "A video of a camera moving" will be used. | None |
| `--output_path`             | Path to where the prepared output samples will be written | `./camera_ctrl_dataset_cached` | |
| `--num_chunks`           | Number of random chunks to extract per zip file                                             | 1 |
| `--height`                 | Height to resize constructed video file                     | 704 |
| `--width`                 |  Width to resize constructed video file                   | 1280 |
| `--num_zip_files`                 | Number of DL3DV/DL3DV-ALL-4K zip files to use for creating the dataset. All used by default                     | -1   |
| `--seed`                 | Seed for randomly selecting chunks from video files                     | 1234 |
| `--tokenizer_dir`                 |  Path to tokenizer dir                   | Downloaded from `nvidia/Cosmos-1.0-Tokenizer-CV8X8X8` to `$HF_HOME` |
