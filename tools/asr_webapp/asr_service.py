# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import atexit
import glob
import os
import shutil
import time
from html import unescape
from uuid import uuid4

import model_api
import torch
import werkzeug
from flask import Flask, make_response, render_template, request, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from nemo.utils import logging

app = Flask(__name__)
CORS(app)

# Upload folder for audio files; models are stored in permanent cache
# which gets deleted once the container shuts down
app.config[f'UPLOAD_FOLDER'] = f"tmp/"


@app.route('/initialize_model', methods=['POST'])
def initialize_model():
    """
    API Endpoint to instantiate a model

    Loads ASR model by its pretrained checkpoint name or upload ASR model that is provided by the user,
    then load that checkpoint into the cache.

    Loading of the model into cache is done once per worker. Number of workers should be limited
    so as not to exhaust the GPU memory available on device (if GPU is being used).
    """
    logging.info("Starting ASR service")
    if torch.cuda.is_available():
        logging.info("CUDA is available. Running on GPU")
    else:
        logging.info("CUDA is not available. Defaulting to CPUs")

    # get form fields
    model_name = request.form['model_names_select']
    use_gpu_if_available = request.form.get('use_gpu_ckbx', "off")

    # get nemo model from user (if not none)
    nemo_model_file = request.files.get('nemo_model', '')

    # if nemo model is not None, upload it to model cache
    if nemo_model_file != '':
        model_name = _store_model(nemo_model_file)

        # Alert user that model has been uploaded into the model cache,
        # and they should refresh the page to access the model
        result = render_template(
            'toast_msg.html', toast_message=f"Model {model_name} has been uploaded. " f"Refresh page !", timeout=5000
        )

    else:
        # Alert user that model has been loaded onto a workers memory
        result = render_template(
            'toast_msg.html', toast_message=f"Model {model_name} has been initialized !", timeout=2000
        )

    # Load model into memory cache
    model_api.initialize_model(model_name=model_name)

    # reset file banner
    reset_nemo_model_file_script = """
        <script>
            document.getElementById('nemo_model_file').value = ""
        </script>
    """

    result = result + reset_nemo_model_file_script
    result = make_response(result)

    # set cookies
    result.set_cookie("model_name", model_name)
    result.set_cookie("use_gpu", use_gpu_if_available)
    return result


def _store_model(nemo_model_file):
    """
    Preserve the model supplied by user into permanent cache
    This cache needs to be manually deleted (if run locally), or gets deleted automatically
    (when the container gets shutdown / killed).

    Args:
        nemo_model_file: User path to .nemo checkpoint.

    Returns:
        A file name (with a .nemo) at the end - to signify this is an uploaded checkpoint.
    """
    filename = secure_filename(nemo_model_file.filename)
    file_basename = os.path.basename(filename)
    model_dir = os.path.splitext(file_basename)[0]

    model_store = os.path.join('models', model_dir)
    if not os.path.exists(model_store):
        os.makedirs(model_store)

    # upload model
    model_path = os.path.join(model_store, filename)
    nemo_model_file.save(model_path)
    return file_basename


@app.route('/upload_audio_files', methods=['POST'])
def upload_audio_files():
    """
    API Endpoint to upload audio files for inference.

    The uploaded files must be wav files, 16 KHz sample rate, mono-channel audio samples.
    """
    # Try to get one or more files from form
    try:
        f = request.files.getlist('file')
    except werkzeug.exceptions.BadRequestKeyError:
        f = None

    # If user did not select any file to upload, notify them.
    if f is None or len(f) == 0:
        toast = render_template('toast_msg.html', toast_message="No file has been selected to upload !", timeout=2000)
        result = render_template('updates/upload_files_failed.html', pre_exec=toast, url=url_for('upload_audio_files'))
        result = unescape(result)
        return result

    # temporary id to store data
    uuid = str(uuid4())
    data_store = os.path.join(app.config[f'UPLOAD_FOLDER'], uuid)

    # If the user attempt to upload another set of files without first transcribing them,
    # delete the old cache of files and create a new cache entirely
    _remove_older_files_if_exists()

    # Save each file into this unique cache
    for fn in f:
        filename = secure_filename(fn.filename)
        if not os.path.exists(data_store):
            os.makedirs(data_store)

        fn.save(os.path.join(data_store, filename))
        logging.info(f"Saving file : {fn.filename}")

    # Update user that N files were uploaded.
    msg = f"{len(f)} file(s) uploaded. Click to upload more !"
    toast = render_template('toast_msg.html', toast_message=f"{len(f)} file(s) uploaded !", timeout=2000)
    result = render_template(
        'updates/upload_files_successful.html', pre_exec=toast, msg=msg, url=url_for('upload_audio_files')
    )
    result = unescape(result)

    result = make_response(result)
    result.set_cookie("uuid", uuid)
    return result


def _remove_older_files_if_exists():
    """
    Helper method to prevent cache leakage when user attempts to upload another set of files
    without first transcribing the files already uploaded.
    """
    # remove old data store (if exists)
    old_uuid = request.cookies.get('uuid', '')
    if old_uuid is not None and old_uuid != '':
        # delete old data store
        old_data_store = os.path.join(app.config[f'UPLOAD_FOLDER'], old_uuid)

        logging.info("Tried uploading more data without using old uploaded data. Purging data cache.")
        shutil.rmtree(old_data_store, ignore_errors=True)


@app.route('/remove_audio_files', methods=['POST'])
def remove_audio_files():
    """
    API Endpoint for removing audio files

    # Note: Sometimes data may persist due to set of circumstances:

        - User uploads audio then closes app without transcribing anything

    In such a case, the files will be deleted when gunicorn shutsdown, or container is stopped.
    However the data may not be automatically deleted if the flast server is used as is.
    """
    # Get the unique cache id from cookie
    uuid = request.cookies.get("uuid", "")
    data_store = os.path.join(app.config[f'UPLOAD_FOLDER'], uuid)

    # If the data does not exist (cache is empty), notify user
    if not os.path.exists(data_store) or uuid == "":
        files_dont_exist = render_template(
            'toast_msg.html', toast_message="No files have been uploaded !", timeout=2000
        )
        result = render_template(
            'updates/remove_files.html', pre_exec=files_dont_exist, url=url_for('remove_audio_files')
        )
        result = unescape(result)
        return result

    else:
        # delete data that exists in cache
        shutil.rmtree(data_store, ignore_errors=True)

        logging.info("Removed all data")

        # Notify user that cache was deleted.
        toast = render_template('toast_msg.html', toast_message="All files removed !", timeout=2000)
        result = render_template('updates/remove_files.html', pre_exec=toast, url=url_for('remove_audio_files'))
        result = unescape(result)

        result = make_response(result)
        result.set_cookie("uuid", '', expires=0)
        return result


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    API Endpoint to transcribe a set of audio files.

    The files are sorted according to their name, so order may not be same as upload order.

    Utilizing the cached info inside the cookies, a model with selected name will be loaded into memory,
    and maybe onto a GPU (if it is supported on the device).

    Then the transcription api will be called from the model_api. If all is successful, a template is updated
    with results. If some issue occurs (memory ran out, file is invalid format), notify the user.
    """
    # load model name from cookie
    model_name = request.cookies.get('model_name')
    logging.info(f"Model name : {model_name}")

    # If model name is not selected via Load Model, notify user.
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

    # If no files found in cache, notify user
    if len(files) == 0:
        result = render_template('toast_msg.html', toast_message="No audio files were found !", timeout=2000)
        return result

    # transcribe file via model api
    t1 = time.time()
    transcriptions = model_api.transcribe_all(files, model_name, use_gpu_if_available=use_gpu_if_available)
    t2 = time.time()

    # delete all transcribed files immediately
    for fp in files:
        try:
            os.remove(fp)
        except FileNotFoundError:
            logging.info(f"Failed to delete transcribed file : {os.path.basename(fp)}")

    # delete temporary transcription directory
    shutil.rmtree(data_store, ignore_errors=True)

    # If something happened during transcription, and it failed, notify user.
    if type(transcriptions) == str and transcriptions == model_api.TAG_ERROR_DURING_TRANSCRIPTION:
        toast = render_template(
            'toast_msg.html',
            toast_message=f"Failed to transcribe files due to unknown reason. "
            f"Please provide 16 KHz Monochannel wav files onle.",
            timeout=5000,
        )
        transcriptions = ["" for _ in range(len(files))]

    else:
        # Transcriptions obtained successfully, notify user.
        toast = render_template(
            'toast_msg.html',
            toast_message=f"Transcribed {len(files)} files using {model_name} (gpu={gpu_used}), "
            f"in {(t2 - t1): 0.2f} s",
            timeout=5000,
        )

    # Write results to data table
    results = []
    for filename, transcript in zip(files, transcriptions):
        results.append(dict(filename=os.path.basename(filename), transcription=transcript))

    result = render_template('transcripts.html', transcripts=results)
    result = toast + result
    result = unescape(result)

    result = make_response(result)
    result.set_cookie("uuid", "", expires=0)
    return result


def remove_tmp_dir_at_exit():
    """
    Helper method to attempt a deletion of audio file cache on flask api exit.
    Gunicorn and Docker container (based on gunicorn) will delete any remaining files on
    shutdown of the gunicorn server or the docker container.

    This is a patch that might not always work for Flask server, but in general should ensure
    that local audio file cache is deleted.

    This does *not* impact the model cache. Flask and Gunicorn servers will *never* delete uploaded models.
    Docker container will delete models *only* when the container is killed (since models are uploaded to
    local storage path inside container).
    """
    try:
        uuid = request.cookies.get("uuid", "")

        if uuid is not None or uuid != "":
            cache_dir = os.path.join(os.path.join(app.config[f'UPLOAD_FOLDER'], uuid))
            logging.info(f"Removing cache file for worker : {os.getpid()}")

            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
                logging.info(f"Deleted tmp folder : {cache_dir}")

    except RuntimeError:
        # Working outside of request context (probably shutdown)
        # simply delete entire tmp folder
        shutil.rmtree(app.config[f'UPLOAD_FOLDER'], ignore_errors=True)


@app.route('/')
def main():
    """
    API Endpoint for ASR Service.
    """
    nemo_model_names, local_model_names = model_api.get_model_names()
    model_names = []
    model_names.extend(local_model_names)  # prioritize local models
    model_names.extend(nemo_model_names)  # attach all other pretrained models

    # page initializations
    result = render_template('main.html', model_names=model_names)
    result = make_response(result)

    # Reset cookies
    result.set_cookie("model_name", '', expires=0)  # model name from pretrained model list
    result.set_cookie("use_gpu", '', expires=0)  # flag to use gpu (if available)
    result.set_cookie("uuid", '', expires=0)  # session id
    return result


# Register hook to delete file cache (for flask server only)
atexit.register(remove_tmp_dir_at_exit)


if __name__ == '__main__':
    app.run(False)
