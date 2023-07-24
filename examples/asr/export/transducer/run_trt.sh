export LD_LIBRARY_PATH="/home/dgalvez/scratch/code/asr/alexa/work/conformer/old_trts/TensorRT-9.0.0.2/lib:/home/dgalvez/scratch/code/asr/alexa/work/conformer/trt-dbg-9.0.0.1/cudnn/lib64:$LD_LIBRARY_PATH"
export PATH="/home/dgalvez/scratch/code/asr/alexa/work/conformer/old_trts/TensorRT-9.0.0.2/bin:$PATH"

# She is running at different batch sizes for some reason...
python3 infer_transducer_trt.py \
    --pretrained_model="stt_en_conformer_transducer_large" \
    --trt_encoder='before_int_change/encoder.trt' \
    --trt_decoder='before_int_change/decoder.trt' \
    --dataset_manifest="/home/dgalvez/scratch/data/test_clean.json" \
    --max_symbold_per_step=5 \
    --batch_size=1 \
    --log


# Largest dimension is 3278 from what I can see...
# batch_size=16 (the original value) does not work.

# [07/19/2023-11:19:11] [TRT] [E] 3: [runtime/api/executionContext.cpp::validateInputBindings::2043] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::validateInputBindings::2043, condition: profileMinDims.d[i] <= dimensions.d[i]. Supplied binding dimension [16,512,0] for bindings[0] exceed min ~ max range at index 2, maximum dimension in profile is 1, minimum dimension in profile is 1, but supplied dimension is 0.
#                                                                                                                                  )
# [07/19/2023-11:19:11] [TRT] [E] 3: [runtime/api/executionContext.cpp::enqueueInternal::740] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::enqueueInternal::740, condition: bindings[x] || nullBindingOK
#                                                                                                                           )
# [07/19/2023-11:19:11] [TRT] [E] 3: [runtime/api/executionContext.cpp::validateInputBindings::2043] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::validateInputBindings::2043, condition: profileMinDims.d[i] <= dimensions.d[i]. Supplied binding dimension [16,512,0] for bindings[0] exceed min ~ max range at index 2, maximum dimension in profile is 1, minimum dimension in profile is 1, but supplied dimension is 0.
#                                                                                                                                  )
# [07/19/2023-11:19:11] [TRT] [E] 3: [runtime/api/executionContext.cpp::enqueueInternal::740] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::enqueueInternal::740, condition: bindings[x] || nullBindingOK
#                                                                                                                           )
# [07/19/2023-11:19:11] [TRT] [E] 3: [runtime/api/executionContext.cpp::validateInputBindings::2043] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::validateInputBindings::2043, condition: profileMinDims.d[i] <= dimensions.d[i]. Supplied binding dimension [16,512,0] for bindings[0] exceed min ~ max range at index 2, maximum dimension in profile is 1, minimum dimension in profile is 1, but supplied dimension is 0.
#                                                                                                                                  )
# [07/19/2023-11:19:11] [TRT] [E] 3: [runtime/api/executionContext.cpp::enqueueInternal::740] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::enqueueInternal::740, condition: bindings[x] || nullBindingOK
#                                                                                                                           )
# [07/19/2023-11:19:11] [TRT] [E] 3: [runtime/api/executionContext.cpp::validateInputBindings::2043] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::validateInputBindings::2043, condition: profileMinDims.d[i] <= dimensions.d[i]. Supplied binding dimension [16,512,0] for bindings[0] exceed min ~ max range at index 2, maximum dimension in profile is 1, minimum dimension in profile is 1, but supplied dimension is 0.
#                                                                                                                                  )

python3 infer_transducer_trt.py \
    --pretrained_model="stt_en_conformer_transducer_large" \
    --trt_encoder='encoder_fp16.trt' \
    --trt_decoder='decoder_fp16.trt' \
    --dataset_manifest="/home/dgalvez/scratch/data/test_clean.json" \
    --max_symbold_per_step=5 \
    --batch_size=16 \
    --log
