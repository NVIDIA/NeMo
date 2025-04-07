Diffusion models
==========================


Data preparation
--------------------------

We expect data to be in this energon dataloader format. For more information about webdataset and energon dataset, please refer to https://github.com/NVIDIA/Megatron-Energon

.pth contains image/video latent representations encode by image/video tokenizer, .json contains metadata including text caption, resolution, aspection ratio, and .pickle contains text embeddings encoded by language model like T5.

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

The following is a sample command to prepare prepare webdataset into energon dataset:

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

