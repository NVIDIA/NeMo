**ASR EVALUATION SERVICE **
===========================

Usage
-----

A simple service that will load a pre-trained model from NeMo's ASR collection and after uploading some files, will transcribe them. Any pre-trained model (or model which has been uploaded to the service) can be selected and if the local / docker environment supports GPUs then it will be possible to transcribe the audio segments with GPU.

All uploaded files are immediately deleted after transcriptions are obtained.

 .. note::

    When using `Gunicorn <https://gunicorn.org/>`_, you might notice that each pretrained checkpoint takes a long time for the first transcription. This is because that worker is instantiating the model into memory and then moving the model onto the GPU. For large models, this step might take a significant amount of time. However, this step is cached, and subsequent transcription requests should be much faster (especially if a GPU is utilized).

From inside the ``asr_webapp`` directory, there are three options to run the ASR Service to transcribe audio -

Local Setup
-----------

1) Install all the dependencies (``pip install -r requirements.txt``). Note : It will utilize the latest branch set in the requirements and will override current nemo installation.

2) Simply run ``python asr_service.py``. This will launch a single service worker that can be run locally.

3) The app will run at ``127.0.0.1:5000`` by default.

Gunicorn Setup
--------------

1) Follow Step (1) of the ``Local Setup`` section.

2) Edit the configuration of ``gunicorn`` inside ``gunicorn.conf.py`` if you wish to change the port or number of workers (though this can be achieved via command line overrides as well).

3) Simply run ``gunicorn wsgi:app``. This will launch two workers (default) in a gunicorn environment.

4) The app will run at ``0.0.0.0:8000`` by default.

Docker Setup
------------

The cleanest approach of the three, and requires simply building and running a docker container.

1) Build the docker by executing ``bash docker_container_build.sh``. This will build a container using the latest branch of nemo.

2) Run the container by executing ``bash docker_container_run.sh``. This will run a detached container that can be used by visiting ``0.0.0.0:8000`` on a modern browser.

Note About Uploading Models
---------------------------

Uploading models is a useful method to evaluate models quickly without having to write scripts.
By design, models uploaded via the browser are visible to **all** users who have access to the web app - so that there only needs to be one user who uploads the model and everyone else who can access this service will be able to evaluate with that model.

This also means that models are uploaded indefinitely - and cannot be removed easily without explicitly deleting the ``models`` directory or by shutting down the container (if a container is used).

There are several reasons this approach was taken -

* It allows for easy evaluation amongst peers without duplicating model uploads (which may be very big).
* When using ``gunicorn``, every worker has access to a central cache of models which can be loaded into memory.
* Model upload mode is meant to be used in a local, protected environment.

If you wish to disable model uploading, please open ``templates/main.html`` and comment out the following section:

.. code-block:: html

    <!-- Load .nemo checkpoint -->
    <!-- Comment this section out to remove ability to upload files -->
    <br>
    NeMo File :
    <input type = "file" name = "nemo_model" accept=".nemo" id="nemo_model_file" />
    <!-- Comment out upto here to remove ability to upload models -->
