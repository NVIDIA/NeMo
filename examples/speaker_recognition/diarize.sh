export HYDRA_FULL_ERROR=1
# speakernet-M-2N-64bs-200e-0.02lr-0.1wr-Vox1Vox2Fisher
EXP_NAMES=" 
speakernet-M-4N-64bs-200e-0.02lr-0.1wr-Vox1Vox2FisherSWBD_energy
"
VAD_JSON="/disk2/callhome_ch109.scp"
EXP_DIR="/data/samsungSSD/NVIDIA/repos/NeMo/examples/speaker_recognition/remote_circe/"
ORACLE_VAD="/disk2/datasets/modified_callhome/modified_oracle_callhome_ch109.json"
ORACLE_MODEL="/disk2/jagadeesh/vad_checkpoints/marblenet-I-4N-64bs-50e-FisherAMI_310ms.nemo"
# ORACLE_VAD="/disk2/datasets/NIST_SRE_2000_LDC2001S97/NIST_SRE_2000_LDC2001S97_16k/modified_oracle_NIST_callhome.json"
# GT_RTTM_DIR="/disk2/datasets/modified_callhome/RTTMS/ch109/"
GT_RTTM_DIR="/disk2/callhome_ch109.rttm"
rm -f result
touch result
reco2num=2
# reco2num="/disk2/datasets/NIST_SRE_2000_LDC2001S97/NIST_SRE_2000_LDC2001S97_16k/reco2num"
for name in $EXP_NAMES;
do
	echo $name
	# result=$(
	python ./speaker_diarize.py \
		diarizer.speaker_embeddings.model_path=$EXP_DIR/$name/spkr.nemo \
		diarizer.path2groundtruth_rttm_files=$GT_RTTM_DIR \
		diarizer.paths2audio_files=$VAD_JSON \
		diarizer.out_dir='/disk2/outputs/diarization/' \
		diarizer.num_speakers=$reco2num \
		diarizer.speaker_embeddings.oracle_vad_manifest=$ORACLE_VAD
	# ) 2> /dev/null || exit 1
	out=$(echo $result | tr ']' '\n' | grep 'Cumulative' | awk '{print $(NF-2),$(NF)}')
	echo $name $out >> result
done

#VAD_JSON="/disk2/callhome_eval.scp"
cat result
# rm result
