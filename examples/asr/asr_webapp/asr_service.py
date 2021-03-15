# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import time

import os
import glob
import shutil
import flask
import torch
import werkzeug
from flask import Flask, json, request, render_template, flash, url_for
from werkzeug.utils import secure_filename

from nemo.utils import logging
import model_api

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = "tmp/"


@app.route('/initialize_model', methods=['POST'])
def initialize_model():
    """
    Loads ASR model
    """
    __MODELS_DICT = None

    logging.info("Starting ASR service")
    if torch.cuda.is_available():
        logging.info("CUDA is available. Running on GPU")
    else:
        logging.info("CUDA is not available. Defaulting to CPUs")

    model_name = request.form['model_names_select']

    model_api.initialize_model(model_name)

    logging.info("ASR service started")
    result = f"Model '{model_name}' has been initialized. Click to reload !"

    return f"""
    <button class="btn mdl-button mdl-js-button mdl-button--raised mdl-button--colored mdl-js-ripple-effect"
            hx-post="{url_for('initialize_model')}" hx-target="this" hx-swap="outerHTML">
        {result}
    </button>
    """


@app.route('/upload_audio_files', methods=['POST'])
def upload_audio_files():
    try:
        f = request.files.getlist('file')
    except werkzeug.exceptions.BadRequestKeyError:
        result = """
        <script>
            alert("No file has been selected to upload !");
        </script>
        """

        return f"""
        {result}
        <button class="btn mdl-button mdl-js-button mdl-button--raised mdl-button--colored mdl-js-ripple-effect"
            hx-post="{url_for('upload_audio_files')}"
            hx-target="this" hx-swap="outerHTML" hx-encoding="multipart/form-data" >
        Upload audio file(s)
        </button>
        """

    for fn in f:
        filename = secure_filename(fn.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        fn.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        logging.info(f"Saving file : {fn.filename}")

    result = f"{len(f)} file(s) uploaded. Click to upload more !"

    return f"""
    <button class="btn mdl-button mdl-js-button mdl-button--raised mdl-button--colored mdl-js-ripple-effect"
            hx-post="{url_for('upload_audio_files')}"
            hx-target="this" hx-swap="outerHTML" hx-encoding="multipart/form-data" >
        {result}
    </button>
    """


@app.route('/remove_audio_files', methods=['POST'])
def remove_audio_files():
    files_dont_exist = """
        <script>
            alert("No files have been uploaded !");
        </script>
        """

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        return f"""
        {files_dont_exist}
        <button class="btn mdl-button mdl-js-button mdl-button--raised mdl-button--colored mdl-js-ripple-effect"
            hx-post="{url_for('remove_audio_files')}"
            hx-target="this" hx-swap="outerHTML">
        Remove all files
        </button>
        """

    else:
        shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER']))
        logging.info("Removed all data")

        result = """
        <script>
            alert("All files removed !");
        </script>
        """

        return f"""
        {result}
        <button class="btn mdl-button mdl-js-button mdl-button--raised mdl-button--colored mdl-js-ripple-effect"
            hx-post="{url_for('remove_audio_files')}"
            hx-target="this" hx-swap="outerHTML">
        Remove all files
        </button>
        """


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not model_api.is_model_availale():
        return """
        <script>
            alert("Model has not been initialized !");
        </script>
        """

    files = list(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], "*.wav")))

    if len(files) == 0:
        return """
        <script>
            alert("No audio files were found !");
        </script>
        """

    transcriptions = model_api.transcribe_all(files)

    results = []
    for filename, transcript in zip(files, transcriptions):
        results.append(dict(filename=os.path.basename(filename), transcription=transcript))
    return render_template('transcripts.html', transcripts=results)


@app.route('/')
def main():
    model_names = sorted(list(model_api.get_model_names()))
    return render_template('main.html', model_names=model_names)


if __name__ == '__main__':
    app.run(debug=True)
