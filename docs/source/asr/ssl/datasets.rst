Datasets
========

LibriLight
----------

The following assume that you already have downloaded Libri-light dataset and segmented into 
audio files of duration uniformly distributed between 32s to 64s and converted into NeMo compatible 
dataset. For information about creating NeMo compatible datasets, refer to the sections 
`Preparing Custom ASR Data`  and `Tarred Datasets` in `ASR dataset<../datasets.rst>`__. To  
improve randomization, further on-the-fly random segments of 32 sec are sampled during SSL 
training. This on-the-fly segmentation is achieved via `random_segment` augmentor with `prob=1` and 
`duration_sec=32` in `train_ds` section of `config` file. Note that the duration of on-the-fly 
segmentation can be scaled according to user resource constraints, however, better results were 
observed for longer durations.
