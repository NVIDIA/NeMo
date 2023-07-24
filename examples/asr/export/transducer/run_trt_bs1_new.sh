export LD_LIBRARY_PATH="/home/dgalvez/scratch/code/asr/alexa/work/conformer/old_trts/TensorRT-9.0.0.2/lib:/home/dgalvez/scratch/code/asr/alexa/work/conformer/trt-dbg-9.0.0.1/cudnn/lib64:$LD_LIBRARY_PATH"
export PATH="/home/dgalvez/scratch/code/asr/alexa/work/conformer/old_trts/TensorRT-9.0.0.2/bin:$PATH"


nsys profile -c cudaProfilerApi \
python3 -u infer_transducer_trt.py \
    --pretrained_model="stt_en_conformer_transducer_large" \
    --trt_encoder='encoder_fp16.trt' \
    --trt_decoder='decoder_fp16.trt' \
    --dataset_manifest="/home/dgalvez/scratch/data/test_clean.json" \
    --max_symbold_per_step=5 \
    --batch_size=1 \
    --log
