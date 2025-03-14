cd tools/ctc_segmentation &&
    TIME=$(date +"%Y-%m-%d-%T") &&
    /bin/bash run_segmentation.sh \
        --MODEL_NAME_OR_PATH=/home/TestData/ctc_segmentation/QuartzNet15x5-Ru-e512-wer14.45.nemo \
        --DATA_DIR=/home/TestData/ctc_segmentation/ru \
        --OUTPUT_DIR=/tmp/ctc_seg_ru/output${TIME} \
        --LANGUAGE=ru \
        --ADDITIONAL_SPLIT_SYMBOLS=";" &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo /home/TestData/ctc_segmentation/verify_alignment.py \
        -r /home/TestData/ctc_segmentation/ru/valid_ru_segments_1.7.txt \
        -g /tmp/ctc_seg_ru/output${TIME}/verified_segments/ru_segments.txt
