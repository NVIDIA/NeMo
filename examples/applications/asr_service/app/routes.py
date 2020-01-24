# Copyright (c) 2019 NVIDIA Corporation
import json
import os
import time

from flask import request
from werkzeug.utils import secure_filename

from app import app, data_preprocessor, jasper_encoder, jasper_decoder, \
    greedy_decoder, neural_factory, MODEL_YAML, WORK_DIR, ENABLE_NGRAM
try:
    from app import beam_search_with_lm
except ImportError:
    print("Not using Beam Search Decoder with LM")
    ENABLE_NGRAM = False
import nemo
import nemo.collections.asr as nemo_asr


def wav_to_text(manifest, greedy=True):
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(MODEL_YAML) as f:
        jasper_model_definition = yaml.load(f)
    labels = jasper_model_definition['labels']

    # Instantiate necessary neural modules
    data_layer = nemo_asr.AudioToTextDataLayer(
        shuffle=False,
        manifest_filepath=manifest,
        labels=labels, batch_size=1)

    # Define inference DAG
    audio_signal, audio_signal_len, _, _ = data_layer()
    processed_signal, processed_signal_len = data_preprocessor(
        input_signal=audio_signal,
        length=audio_signal_len)
    encoded, encoded_len = jasper_encoder(audio_signal=processed_signal,
                                          length=processed_signal_len)
    log_probs = jasper_decoder(encoder_output=encoded)
    predictions = greedy_decoder(log_probs=log_probs)

    if ENABLE_NGRAM:
        print('Running with beam search')
        beam_predictions = beam_search_with_lm(
            log_probs=log_probs, log_probs_length=encoded_len)
        eval_tensors = [beam_predictions]

    if greedy:
        eval_tensors = [predictions]

    tensors = neural_factory.infer(tensors=eval_tensors)
    if greedy:
        from nemo.collections.asr.helpers import post_process_predictions
        prediction = post_process_predictions(tensors[0], labels)
    else:
        prediction = tensors[0][0][0][0][1]
    return prediction


result_template = """
<html>
<h3 align="center">Transcription Result</h3>
   <body style="border:3px solid green">
   <div align="center">
   <p>Transcription time: {0}</p>
   <p>{1}</p>
   </div>
   </body>
</html>
"""


@app.route('/transcribe_file', methods=['GET', 'POST'])
def transcribe_file():
    if request.method == 'POST':
        # upload wav_file to work directory
        f = request.files['file']
        greedy = True
        if request.form.get('beam'):
            if not ENABLE_NGRAM:
                return ("Error: Beam Search with ngram LM is not enabled "
                        "on this server")
            greedy = False
        file_path = os.path.join(WORK_DIR, secure_filename(f.filename))
        f.save(file_path)
        # create manifest
        manifest = dict()
        manifest['audio_filepath'] = file_path
        manifest['duration'] = 18000
        manifest['text'] = 'todo'
        with open(file_path+".json", 'w') as fout:
            fout.write(json.dumps(manifest))
        start_t = time.time()
        transcription = wav_to_text(file_path + ".json", greedy=greedy)
        total_t = time.time() - start_t
        result = result_template.format(total_t, transcription)
        return str(result)


@app.route('/')
@app.route('/index')
def index():
    return "Hello from NeMo ASR webservice!"
