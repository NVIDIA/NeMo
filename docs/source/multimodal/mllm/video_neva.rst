Video NeVA
==========

Model Introduction
------------------

Video NeVa adds support for video modality in NeVa by representing video as multiple image frames.

Video Neva Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

  data:
    media_type: video
    splice_single_frame: null
    num_frames: 8
    image_folder: null
    video_folder: null

- ``media_type``: If set to `video`, NeVa's dataloader goes through the additional preprocessing steps to represent the input video data as a series of image frames.
- ``splice_single_frame``: Can either be set as `first`, `middle` or `last`. This will result in only a single frame in that specific location of the video being selected.
- ``num_frames``: This is used to select the number of image frames that will be used to represent the video.
- ``video_folder``: This specifies the directory where the video files are located. This follows the same format as NeVa's `image_folder`.

References
----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS
    :keyprefix: mm-models-
