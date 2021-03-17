**ASR SERVICE **
================

Usage
-----

A simple service that will load a pre-trained model from NeMo's ASR collection and after uploading some files, will transcribe them. Any pre-trained model can be selected and if the local / docker environment supports GPUs then it will be possible to transcribe the audio segments with GPU.

Note: All uploaded files are immediately deleted after transcriptions are obtained.

There are three options to run the ASR Service to transcribe audio -

Local Setup
-----------

1) Install all the dependencies (``pip install -r requirements.txt``). Note : It will utilize the latest branch set in the requirements and will override current nemo installation.

2) Simply run ``python asr_service.py``. This will launch a single service worker that can be run locally.

3) The app will run at ``127.0.0.1:5000`` by default.

Gunicorn Setup
--------------

1) Follow above steps for ``Local Setup``

2) Edit the configuration of ``gunicorn`` inside ``gunicorn.conf.py`` if you wish to change the port or number of workers (though this can be achieved via command line overrides as well).

3) Simply run ``gunicorn wsgi:app``. This will launch two workers (default) in a gunicorn environment.

4) The app will run at ``0.0.0.0:8000`` by default.

Docker Setup
------------

The cleanest approach of the three, and requires simply building and running a docker container.

1) Build the docker by executing ``bash docker_container_build.sh``. This will build a container using the latest branch of nemo.

2) Run the container by executing ``bash docker_container_run.sh``. This will run a detached container that can be used by visiting ``0.0.0.0:8000`` on a modern browser.
