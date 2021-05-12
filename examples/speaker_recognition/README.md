# Speaker Recognition

Documentation section for speaker related tasks can be found at:
 - [Speaker Recognition and Verification](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html)
 - [Speaker Diarization](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html)

This folder contains scripts for performing various speaker related tasks as breiefly explained below:
 - **speaker_reco.py**
: Training script for speaker verification and recognition
 - **speaker_reco_infer.py**
: Inference for speaker recognition. Can be used to map labels using trained model on test set with known speaker labels from train set. (speaker id) 
 - **extract_speaker_embeddings.py**
: Extract speaker embeddings using trained model for speaker verification purposes. 
 - **speaker_reco_finetune.py**
: Sample script to demonstrate on how to finetune a already trained model for specific sub tasks.
 - **voxceleb_eval.py**
: Helper script to evaluate speaker embeddings on voxceleb trail files.  
 - **speaker_diarize.py**
: Inference script for speaker diarization - knowing who spoke when. 
 - **conf/**
: Folder containing configuration files for training speaker tasks. We currently only support inference for speaker diarization. 
