#!/bin/bash
python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    --asr_model=/models/sel_ngcinit_nemoasrset3.0_d512_adamwlr2.0_wd0_augx_speunigram1024_streaming_104_12_wm10k_ctc_striding4x_400e_clip1_newcode.nemo \
    --manifest_file=/datasets/ls_test_other/transcripts.local.json \
    --batch_size=128

