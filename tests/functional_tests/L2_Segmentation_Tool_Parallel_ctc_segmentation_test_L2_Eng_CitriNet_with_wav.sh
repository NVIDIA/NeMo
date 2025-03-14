cd tools/ctc_segmentation &&
    TIME=$(date +"%Y-%m-%d-%T") &&
    /bin/bash run_segmentation.sh \
        --MODEL_NAME_OR_PATH="stt_en_citrinet_512_gamma_0_25" \
        --DATA_DIR=/home/TestData/ctc_segmentation/eng \
        --OUTPUT_DIR=/tmp/ctc_seg_en/output${TIME} \
        --LANGUAGE=en \
        --USE_NEMO_NORMALIZATION="TRUE" &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo /home/TestData/ctc_segmentation/verify_alignment.py \
        -r /home/TestData/ctc_segmentation/eng/eng_valid_segments_1.7.txt \
        -g /tmp/ctc_seg_en/output${TIME}/verified_segments/nv_test_segments.txt
