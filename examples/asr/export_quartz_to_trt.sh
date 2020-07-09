#!/bin/bash
#
# Export models to serialized TensorRT engine plan files
# After models have been downloaded, run this from project root:
#
#   $ ./scripts/export_quartz_to_trt.sh 
#
#!/bin/bash
#set -Eueo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/../.."

ONNX_TO_TRT="${DIR}/scripts/export_jasper_onnx_to_trt.py"
JASPER_REPO="${DIR}/examples/asr/triton/repository/jasper-asr-streaming-vad"


MODEL_DIR=${DIR}/examples/asr/models/quartznet15x5
mkdir -p ${MODEL_DIR}

CHECKPOINT_ENCODER="JasperEncoder-STEP-243800.pt"
CHECKPOINT_DECODER="JasperDecoderForCTC-STEP-243800.pt"

if [ ! -f ${MODEL_DIR}/${CHECKPOINT_ENCODER} ]; then
    wget https://api.ngc.nvidia.com/v2/models/nvidia/multidataset_quartznet15x5/versions/1/files/${CHECKPOINT_ENCODER} -P ${MODEL_DIR}
fi

if [ ! -f ${MODEL_DIR}/${CHECKPOINT_DECODER} ]; then
    wget https://api.ngc.nvidia.com/v2/models/nvidia/multidataset_quartznet15x5/versions/1/files/${CHECKPOINT_DECODER} -P ${MODEL_DIR}
fi

echo "exporting ASR models to ONNX ..."

python3 ${DIR}/scripts/export_jasper_to_onnx.py --config ${DIR}/examples/asr/configs/quartznet15x5.yaml --nn_decoder ${MODEL_DIR}/${CHECKPOINT_DECODER} --nn_encoder ${MODEL_DIR}/${CHECKPOINT_ENCODER} --onnx_decoder ${JASPER_REPO}/jasper-onnx-decoder-streaming/1/nn_decoder.onnx --onnx_encoder ${JASPER_REPO}/jasper-onnx-encoder-streaming/1/nn_encoder.onnx

echo "exporting QuartzNet-v1 encoder.onnx to TensorRT engine..."
python3 $ONNX_TO_TRT ${JASPER_REPO}/jasper-onnx-encoder-streaming/1/nn_encoder.onnx ${JASPER_REPO}/jasper-trt-encoder-streaming/1/model.plan --max-seq-len 251 --seq-len 251 --batch-size 1 --max-batch-size 1  --max-workspace-size 128
echo "done exporting QuartzNet-v1 encoder.onnx to TensorRT engine..."

echo "exporting QuartzNet-v1 decoder.onnx to TensorRT engine..."
python3 $ONNX_TO_TRT ${JASPER_REPO}/jasper-onnx-decoder-streaming/1/nn_decoder.onnx ${JASPER_REPO}/jasper-trt-decoder-streaming/1/model.plan --max-seq-len 126 --seq-len 126 --batch-size 1 --max-batch-size 1 --max-workspace-size 16
echo "done exporting QuartzNet-v1 decoder.onnx to TensorRT engine"


