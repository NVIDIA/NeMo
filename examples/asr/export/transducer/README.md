## Setup

Please install nemo and tensorrt.

The below test passes through tensorrt ngc docker container.

For nemo installation, please refer to [here](https://github.com/NVIDIA/NeMo/#installation).
For tensorrt 9.x installation, please download packages from [here](http://cuda-repo/release-candidates/Libraries/TensorRT/v9.0/)
 and install according to the packages you downloaded.

Clone nemo project:
```
git clone https://github.com/NVIDIA/NeMo.git
mv * NeMo/examples/asr/export/transducer/
cd NeMo/examples/asr/export/transducer/
```


Please run the below command to generate onnx model:
```
bash run.sh
```

Be sure to prepare `test.json` file.

Please run the below command to convert the generated onnx model:
```
bash convert.sh
```

Please run the below command to do inference:
```
bash run_trt.sh
```
