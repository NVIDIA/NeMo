Automatic Speech Recognition (ASR)
==================================

Automatic Speech Recognition (ASR), also known as Speech To Text (STT), refers to the problem of automatically transcribing spoken language.
You can use NeMo to transcribe speech using open-sourced pretrained models in 14+ languages [link], or train your own ASR models[link].

Transcribe speech with 3 lines of code
----------------------------------------
After installing NeMo [link], you can transcribe an audio file as follows:

.. code-block:: python

    import nemo.collections.asr as nemo_asr
    asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_transducer_large")
    transcript = asr_model.transcribe(["path/to/audio_file.wav"])

Obtain word timestamps
^^^^^^^^^^^^^^^^^^^^^^^^^

You can also obtain timestamps for each word in the transcription by adding a flag ``return_hypotheses=True``:

.. code-block:: python

    # import nemo_asr and instantiate asr_model as above
    import nemo.collections.asr as nemo_asr
    asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_transducer_large")
    # specify flag `return_hypotheses=True``
    transcript_and_timestamps = asr_model.transcribe(["path/to/audio_file.wav"], return_hypotheses=True)

Transcribe speech via command line
----------------------------------
You can also transcribe speech via the command line using the following `script <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py>`_, for example:

.. code-block:: bash

    python <path_to_NeMo>/blob/main/examples/asr/transcribe_speech.py \
        pretrained_name="stt_en_fastconformer_transducer_large" \
        audio_dir=<path_to_audio_dir> # path to dir containing audio files to transcribe

The script will save all transcriptions in a JSONL file where each line corresponds to an audio file in ``<audio_dir>``.
This file will correspond to a format that NeMo commonly uses for saving model predictions, and also for storing 
input data for training and evaluation. You can learn more about the format that NeMo uses for these files 
(which we refer to as "manifest files") here [link].

You can also specify the files to be transcribed inside a manifest file, and pass that in using the argument 
``dataset_manifest=<path to manifest specifying audio files to transcribe>`` instead of ``audio_dir``.


Incorporate a language model (LM) to improve ASR transcriptions
---------------------------------------------------------------

You can often get a boost to transcription accuracy by using a Language Model to help choose words that are more likely
to be spoken in a sentence.

You can get a good improvment in transcription accuracy even using a simple N-gram LM.

After training an N-gram LM [link], or using one provided by NeMo[link], you can incorporate it into your ASR model as follows:

1. Install the OpenSeq2Seq beam search decoding and KenLM libraries using `this script <scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh>`_.
2. Perform transcription using `this script <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py>`_:

.. code-block:: bash

    python eval_beamsearch_ngram.py nemo_model_file=<path to the .nemo file of the model> \
    input_manifest=<path to the evaluation JSON manifest file \
    kenlm_model_file=<path to the binary KenLM model> \
    beam_width=[<list of the beam widths, separated with commas>] \
    beam_alpha=[<list of the beam alphas, separated with commas>] \
    beam_beta=[<list of the beam betas, separated with commas>] \
    preds_output_folder=<optional folder to store the predictions> \
    probs_cache_file=null \
    decoding_mode=beamsearch_ngram \
    decoding_strategy="<Beam library such as beam, pyctcdecode or flashlight>"

[ todo: simplify command and/or explain args]

Use real-time transcription
---------------------------

It is possible to use NeMo to transcribe speech in real-time. You can find an example of how to do 
this in the following `notebook tutorial <https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo.ipynb>`_.


Try different ASR models
------------------------


Try out NeMo ASR models in your browser
---------------------------------------
You can try out the NeMo ASR model transcriptions without leaving your browser using the HuggingFace Space embedded below.

.. raw:: html

    <iframe src="https://hf.space/embed/smajumdar/nemo_multilingual_language_id/+"
    width="100%" class="gradio-asr" allow="microphone *"></iframe>

    <script type="text/javascript" language="javascript">
        $('.gradio-asr').css('height', $(window).height()+'px');
    </script>


The full documentation tree is as follows:

.. toctree::
   :maxdepth: 8

   models
   datasets
   asr_language_modeling
   results
   scores
   configs
   api
   resources
   examples/kinyarwanda_asr.rst

.. include:: resources.rst
