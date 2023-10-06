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
------------------------------------


Incorporate a language model (LM) to improve ASR transcriptions
-----------------------------------------------------------------


Use real-time transcription
---------------------------



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
