Video NeVA
==========

Model Introduction
------------------

Video NeVa adds support for video modality in NeVa by representing video as multiple image frames. 

There is only a minor change done to :class:`~nemo.collections.multimodal.models.multimodal_llm.neva.neva_model.MegatronNevaModel` class in order to support pretraining on video input data.

Representing video input as a series of images is done in :class:`~nemo.collections.multimodal.data.neva.TarOrFolderVideoLoader` class, using Decord which provides convenient video slicing methods. 


Video Neva Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

  data:
    media_type: video
    splice_single_frame: null
    num_frames: 8
    image_token_len: 256
    image_folder: null
    video_folder: null

- ``media_type``: If set to `video`, NeVa's dataloader goes through the additional preprocessing steps to represent the input video data as a series of image frames.
- ``splice_single_frame``: Can either be set as `first`, `middle` or `last`. This will result in only a single frame in that specific location of the video being selected.
- ``image_token_len``: The NeVa dataloader calculates `image_token_len` based on the height and width of the preprocessed image frame and the patch size of the CLIP model being used. 
.. code-block:: python

image_token_len = (224 // 14) * (224 // 14) = 16 * 16 = 256

- ``num_frames``: This is used to select the number of image frames that will be used to represent the video.
- ``video_folder``: This specifies the directory where the video files are located. This follows the same format as NeVa's `image_folder`.

References
----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS
    :keyprefix: mm-models-
