Contains converters to the [RLDS format](https://github.com/google-research/rlds), which is a specification on top of the [TFDS](https://www.tensorflow.org/datasets) (TensorFlow datasets) format, which is for the most part built on top of the TFRecord format. RLDS datasets can be loaded using `dlimp.DLataset.from_rlds`.

Out of the box, TFDS only supports single-threaded dataset conversion and distributed dataset conversion using Apache Beam. `dataset_builder.py` contains a more middle-ground implementation that uses Python multiprocessing to parallelize conversion on a single machine. It is based heavily on Karl Pertsch's implementation (see [kpertsch/bridge_rlds_builder](https://github.com/kpertsch/bridge_rlds_builder/blob/f0d16c5a8384c1476aa1c274a9aef3a5f76cbada/bridge_dataset/conversion_utils.py)).

## Usage
Each subdirectory contains a specific dataset converter implementation that inherits from the `dataset_builder.MultiThreadedDatasetBuilder` class. First, install the multithreaded dataset builder by running `pip install .` in this directory. Each dataset converter may have additional requirements that they specify using a `requirements.txt`.

To build a particular dataset, `cd` into its corresponding directory and run `CUDA_VISIBLE_DEVICES="" tfds build --manual_dir <path_to_raw_data>`. See individual dataset documentation for how to obtain the raw data. You may also want to modify settings inside the `<dataset_name>_dataset_builder.py` file (e.g., `NUM_WORKERS` and `CHUNKSIZE`.)
