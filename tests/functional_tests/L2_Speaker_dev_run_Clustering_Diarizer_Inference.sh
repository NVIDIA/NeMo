coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/speaker_tasks/diarization/clustering_diarizer/offline_diar_infer.py \
    diarizer.manifest_filepath=/home/TestData/an4_diarizer/an4_manifest.json \
    diarizer.speaker_embeddings.model_path=/home/TestData/an4_diarizer/spkr.nemo \
    diarizer.speaker_embeddings.parameters.save_embeddings=True \
    diarizer.speaker_embeddings.parameters.window_length_in_sec=1.5 \
    diarizer.speaker_embeddings.parameters.shift_length_in_sec=0.75 \
    diarizer.speaker_embeddings.parameters.multiscale_weights=null \
    diarizer.vad.model_path=/home/TestData/an4_diarizer/MatchboxNet_VAD_3x2.nemo \
    diarizer.out_dir=/tmp/clustering_diarizer_results
