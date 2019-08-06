Jasper Speech Recognizer Service
================================
This example is intended to get you started quickly for experimentation with ASR service powered by Jasper.

*This is probably not how you should serve things in production*
*Note: the service will only work correctly with single channel 16Khz .wav files*.

The example consists of two parts:

1) ``recognize.html`` - trivial HTML form which will upload .wav file to ASR service
2) Flask-based ``ASR service`` which accepts .wav file and returns it's transcription

To get started
~~~~~~~~~~~~~~

1) Install Flask: ``pip install flask```
2) Create WORKDIR folder (anywhere) to be used in (3)
3) In the file ``<nemo_git_root>/examples/applications/asr_service/app/__init__.py`` modify `WORK_DIR`, `MODEL_YAML`, `CHECKPOINT_ENCODER` and `CHECKPOINT_DECODER` to point to the correct values
5) From `<nemo_git_root>/examples/applications/asr_service` folder do: `export FLASK_APP=asr_service.py` and start service: `flask run --host=0.0.0.0`
6) Modify `recognize.html`: replace `<flask_service_ip>` with the IP address of machine where flask service from Step 5 is running.
7) Open `recognize.html` with any browser and upload a .wav file

You can also enable BeamSearch with KenLM language model. Set `ENABLE_NGRAM=True` in `examples/applications/asr_service/app/__init__.py` to enable running with BeamSearch and KenLM.
Also you must install Baidu's CTC decoder and KenLM. Do do so (with KenLM built on LibriSpeech dataset) do:

*Note: building 6-gram KenLM on LibriSpeech data will take ~2 hours*.

* cd ``<nemo_git_root>/scripts``
* `./install_decoders.sh`
* `./build_6-gram_OpenSLR_lm.sh`





