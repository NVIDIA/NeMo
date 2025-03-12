python examples/speaker_tasks/diarization/neural_diarizer/multiscale_diar_decoder_infer.py \
diarizer.manifest_filepath=/home/TestData/an4_diarizer/an4_manifest.json \
diarizer.msdd_model.model_path=/home/TestData/an4_diarizer/diar_msdd_telephonic.nemo \
diarizer.speaker_embeddings.parameters.save_embeddings=True \
diarizer.vad.model_path=/home/TestData/an4_diarizer/MatchboxNet_VAD_3x2.nemo \
diarizer.out_dir=/tmp/neural_diarizer_results
