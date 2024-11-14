Diffusion Dataset and dataloader
===========================

Dataloader 
------------
1. `<diffusion_energon_datamodule.py>`_ : webdataset loader
2. `<diffusion_fake_datamodule.py>`_ : mock data loader

Preparing Image / Video Megatron Energon WebDataset with Cosmos Tokenizer
------------

We expect data to be in this webdataset format. For more information about webdataset and energon dataset, please refer to https://github.com/NVIDIA/Megatron-Energon

This script is an example on preparing a WebDataset for an image / video + text dataset using distributed processing with the Cosmos Tokenizer. It processes each sample by generating a **continuous** image / video latent using the Cosmos video tokenizer and a T5 embedding from the text caption. Then, the processed data is stored in a WebDataset-compatible format.

Requirements
^^^^^^^^^^^^^^^
- **Dependencies**:
  - Please use the latest NeMo dev container: ``nvcr.io/nvidia/nemo:dev``
  - You may also need to install ``jammy`` and ``mediapy`` depending on your dev container version.

- **Data**:
  - The script uses an example dataset that comes in parquet format. To use a custom, you will need to write a custom ``process_func`` and create a new factory recipe that uses your new ``process_func``.

Usage
^^^^^^^^^^^^^^^
1. **Set up your environment**:
   Pull and launch the NeMo dev container to run your script.

2. **Customize Cache Path**:
   Set the T5 cache directory path in the script by specifying the `t5_cache_dir` variable.

3. **Running the Script**:
   To run the script on 8 GPUs, use the following command:
   
   ``bash torchrun --nproc_per_node=8 nemo/collections/diffusion/data/prepare_energon_dataset.py``

4. **Generate meta data for the dataset**:
    The above script will generate a folder a tar files. .pth contains image/video latent representations encode by image/video tokenizer, .json contains metadata including text caption, resolution, aspection ratio, and .pickle contains text embeddings encoded by language model like T5.
    
    .. code-block:: bash
    
       shard_000.tar
       ├── samples/sample_0000.pth
       ├── samples/sample_0000.pickle
       ├── samples/sample_0000.json
       ├── samples/sample_0001.pth
       ├── samples/sample_0001.pickle
       ├── samples/sample_0001.json
       └── ...
       shard_001.tar   
    
    Below we generate energon metadata for the webdataset.
    
    .. code-block:: bash
    
       # energon prepare . --num-workers 192
       Found 369057 tar files in total. The first and last ones are:
       - 0.tar
       - 99999.tar
       If you want to exclude some of them, cancel with ctrl+c and specify an exclude filter in the command line.
       Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 1,0,0
       Indexing shards  [####################################]  369057/369057
       Sample 0, keys:
       - .json
       - .pickle
       - .pth
       Sample 1, keys:
       - .json
       - .pickle
       - .pth
       Found the following part types in the dataset: .json, .pth, .pickle
       Do you want to create a dataset.yaml interactively? [Y/n]: Y
       The following dataset classes are available:
       0. CaptioningWebdataset
       1. CrudeWebdataset
       2. ImageClassificationWebdataset
       3. ImageWebdataset
       4. InterleavedWebdataset
       5. MultiChoiceVQAWebdataset
       6. OCRWebdataset
       7. SimilarityInterleavedWebdataset
       8. TextWebdataset
       9. VQAOCRWebdataset
       10. VQAWebdataset
       11. VidQAWebdataset
       Please enter a number to choose a class: 1
       The dataset you selected uses the following sample type:
    
       class CrudeSample(dict):
          """Generic sample type to be processed later."""
    
       CrudeWebdataset does not need a field map. You will need to provide a `Cooker` for your dataset samples in your `TaskEncoder`.
       Furthermore, you might want to add `subflavors` in your meta dataset specification.
5. in training, you can specify the path to dataset using ``data.path=path/to/dataset`` in command line or ``train.py``.
