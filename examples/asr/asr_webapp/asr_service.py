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
import time

from uuid import uuid4
from flask import Flask, json, request, render_template, url_for, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from html import unescape

from nemo.utils import logging
import model_api

app = Flask(__name__)
CORS(app)

app.config[f'UPLOAD_FOLDER'] = f"tmp/"


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
    use_gpu_if_available = request.form.get('use_gpu_ckbx', "off")

    result = render_template('toast_msg.html',
                             toast_message=f"Model {model_name} has been initialized !",
                             timeout=2000)
    result = make_response(result)

    # set cookies
    result.set_cookie("model_name", model_name)
    result.set_cookie("use_gpu", use_gpu_if_available)
    return result


@app.route('/upload_audio_files', methods=['POST'])
def upload_audio_files():
    try:
        f = request.files.getlist('file')
    except werkzeug.exceptions.BadRequestKeyError:
        f = None

    if f is None or len(f) == 0:
        toast = render_template('toast_msg.html',
                                toast_message="No file has been selected to upload !",
                                timeout=2000)
        result = render_template('updates/upload_files_failed.html', pre_exec=toast, url=url_for('upload_audio_files'))
        result = unescape(result)
        return result

    # temporary id to store data
    uuid = str(uuid4())
    data_store = os.path.join(app.config[f'UPLOAD_FOLDER'], uuid)

    # remove old data store (if exists)
    old_uuid = request.cookies.get('uuid', '')
    if old_uuid is not None and old_uuid != '':
        # delete old data store
        old_data_store = os.path.join(app.config[f'UPLOAD_FOLDER'], old_uuid)

        logging.info("Tried uploading more data without using old uploaded data. Purging data cache.")
        shutil.rmtree(old_data_store, ignore_errors=True)

    for fn in f:
        filename = secure_filename(fn.filename)
        if not os.path.exists(data_store):
            os.makedirs(data_store)

        fn.save(os.path.join(data_store, filename))
        logging.info(f"Saving file : {fn.filename}")

    msg = f"{len(f)} file(s) uploaded. Click to upload more !"
    toast = render_template('toast_msg.html', toast_message=f"{len(f)} file(s) uploaded !", timeout=2000)
    result = render_template(
        'updates/upload_files_successful.html', pre_exec=toast, msg=msg, url=url_for('upload_audio_files')
    )
    result = unescape(result)

    result = make_response(result)
    result.set_cookie("uuid", uuid)
    return result


@app.route('/remove_audio_files', methods=['POST'])
def remove_audio_files():
    uuid = request.cookies.get("uuid", "")
    data_store = os.path.join(app.config[f'UPLOAD_FOLDER'], uuid)

    if not os.path.exists(data_store) or uuid == "":
        files_dont_exist = render_template('toast_msg.html', toast_message="No files have been uploaded !", timeout=2000)
        result = render_template(
            'updates/remove_files.html', pre_exec=files_dont_exist, url=url_for('remove_audio_files')
        )
        result = unescape(result)
        return result

    else:
        # delete data
        shutil.rmtree(data_store, ignore_errors=True)

        logging.info("Removed all data")

        toast = render_template('toast_msg.html', toast_message="All files removed !", timeout=2000)
        result = render_template('updates/remove_files.html', pre_exec=toast, url=url_for('remove_audio_files'))
        result = unescape(result)

        result = make_response(result)
        result.set_cookie("uuid", '', expires=0)
        return result


@app.route('/transcribe', methods=['POST'])
def transcribe():
    # load model name from cookie
    model_name = request.cookies.get('model_name')
    logging.info(f"Model name : {model_name}")

    if model_name is None or model_name == '':
        result = render_template('toast_msg.html', toast_message="Model has not been initialized !", timeout=2000)
        return result

    # load whether gpu should be used
    use_gpu_if_available = request.cookies.get('use_gpu') == 'on'
    gpu_used = torch.cuda.is_available() and use_gpu_if_available

    # Load audio from paths
    uuid = request.cookies.get("uuid", "")
    data_store = os.path.join(app.config[f'UPLOAD_FOLDER'], uuid)

    files = list(glob.glob(os.path.join(data_store, "*.wav")))

    if len(files) == 0:
        result = render_template('toast_msg.html', toast_message="No audio files were found !", timeout=2000)
        return result

    # transcribe file
    t1 = time.time()
    transcriptions = model_api.transcribe_all(files, model_name, use_gpu_if_available)
    t2 = time.time()

    # delete all transcribed files immediately
    for fp in files:
        try:
            os.remove(fp)
        except FileNotFoundError:
            logging.info(f"Failed to delete transcribed file : {os.path.basename(fp)}")

    # delete temporary transcription directory
    shutil.rmtree(data_store, ignore_errors=True)

    # Write results to table
    results = []
    for filename, transcript in zip(files, transcriptions):
        results.append(dict(filename=os.path.basename(filename), transcription=transcript))

    toast = render_template(
        'toast_msg.html',
        toast_message=f"Transcribed {len(files)} files using {model_name} (gpu={gpu_used}), "
        f"in {(t2 - t1): 0.2f} s",
        timeout=10000
    )
    result = render_template('transcripts.html', transcripts=results)
    result = toast + result
    result = unescape(result)

    result = make_response(result)
    result.set_cookie("uuid", "", expires=0)
    return result


def remove_tmp_dir_at_exit():
    uuid = request.cookies.get("uuid", "")

    if uuid is not None or uuid != "":
        cache_dir = os.path.join(os.path.join(app.config[f'UPLOAD_FOLDER'], uuid))
        logging.info(f"Removing cache file for worker : {os.getpid()}")

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            logging.info(f"Deleted tmp folder : {cache_dir}")


@app.route('/')
def main():
    model_names = sorted(list(model_api.get_model_names()))

    # page initializations
    result = render_template('main.html', model_names=model_names)
    result = make_response(result)

    # Reset cookies
    result.set_cookie("model_name", '', expires=0)
    result.set_cookie("use_gpu", '', expires=0)
    result.set_cookie("uuid", '', expires=0)
    return result


# Register hook to delete file cache
atexit.register(remove_tmp_dir_at_exit)


if __name__ == '__main__':
    app.run(debug=True)
