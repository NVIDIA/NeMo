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

import os
import glob
import shutil
import torch
import werkzeug
import atexit

from flask import Flask, json, request, render_template, url_for
from werkzeug.utils import secure_filename
from html import unescape

from nemo.utils import logging
import model_api

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = f"tmp_{os.getpid()}/"


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

    result = render_template('toast_msg.html', toast_message=f"Model {model_name} has been initialized !")
    return result


@app.route('/upload_audio_files', methods=['POST'])
def upload_audio_files():
    try:
        f = request.files.getlist('file')
    except werkzeug.exceptions.BadRequestKeyError:
        f = None

    if f is None or len(f) == 0:
        toast = render_template('toast_msg.html', toast_message="No file has been selected to upload !")
        result = render_template('updates/upload_files_failed.html', pre_exec=toast,
                                 url=url_for('upload_audio_files'))
        result = unescape(result)
        return result

    for fn in f:
        filename = secure_filename(fn.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        fn.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        logging.info(f"Saving file : {fn.filename}")

    msg = f"{len(f)} file(s) uploaded. Click to upload more !"
    toast = render_template('toast_msg.html', toast_message=f"{len(f)} file(s) uploaded !")
    result = render_template('updates/upload_files_successful.html',
                             pre_exec=toast,
                             msg=msg,
                             url=url_for('upload_audio_files'))
    result = unescape(result)
    return result


@app.route('/remove_audio_files', methods=['POST'])
def remove_audio_files():

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        files_dont_exist = render_template('toast_msg.html', toast_message="No files have been uploaded !")
        result = render_template('updates/remove_files.html', pre_exec=files_dont_exist,
                                 url=url_for('remove_audio_files'))
        result = unescape(result)
        return result

    else:
        shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER']))
        logging.info("Removed all data")

        toast = render_template('toast_msg.html', toast_message="All files removed !")
        result = render_template('updates/remove_files.html', pre_exec=toast,
                                 url=url_for('remove_audio_files'))
        result = unescape(result)
        return result


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not model_api.is_model_availale():
        result = render_template('toast_msg.html', toast_message="Model has not been initialized !")
        return result

    files = list(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], "*.wav")))

    if len(files) == 0:
        result = render_template('toast_msg.html', toast_message="No audio files were found !")
        return result

    transcriptions = model_api.transcribe_all(files)

    results = []
    for filename, transcript in zip(files, transcriptions):
        results.append(dict(filename=os.path.basename(filename), transcription=transcript))
    return render_template('transcripts.html', transcripts=results)


def remove_tmp_dir_at_exit():
    cache_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        logging.info(f"Deleted tmp folder : {cache_dir}")


@app.route('/')
def main():
    model_names = sorted(list(model_api.get_model_names()))

    # button initializations
    return render_template('main.html', model_names=model_names)


# Register hook to delete file cache
atexit.register(remove_tmp_dir_at_exit)


if __name__ == '__main__':
    app.run(debug=True)
