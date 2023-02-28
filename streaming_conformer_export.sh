#!/bin/bash -x

python scripts/export.py \
	--onnx-opset=16 \
	--verbose debug \
	--check-tolerance 1.0 \
	--runtime-check \
	--streaming_support \
	--cache_support /models/sel_ngcinit_nemoasrset3.0_d512_adamwlr2.0_wd0_augx_speunigram1024_streaming_104_12_wm10k_ctc_striding4x_400e_clip1_newcode.nemo \
	streaming-conformer.onnx > export.log 2>&1
