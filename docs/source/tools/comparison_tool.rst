Comparison tool for ASR Models 
====================

Comparison tool is integrated in SDE feature, that allows to visualize predictions of different models.

+--------------------------------------------------------------------------------------------------------------------------+
| **Comparation tool features:**                                                                                                        |
+--------------------------------------------------------------------------------------------------------------------------+
| navigation across the dataset using an interactive datatable that supports sorting and filtering                         |
+--------------------------------------------------------------------------------------------------------------------------+
| inspection of individual words on the plot             |
+--------------------------------------------------------------------------------------------------------------------------+
| Corner cases detection                                |
+--------------------------------------------------------------------------------------------------------------------------+
| Visual Comparison of Two Model Predictions                                           |
+--------------------------------------------------------------------------------------------------------------------------+

Getting Started
---------------
As comparison tool integrated in SDE, it could be found at `NeMo/tools/speech_data_explorer <https://github.com/NVIDIA/NeMo/tree/main/tools/speech_data_explorer>`__.

Please install the SDE requirements:

.. code-block:: bash

    pip install -r tools/speech_data_explorer/requirements.txt

Then run:

.. code-block:: bash

    python tools/speech_data_explorer/data_explorer.py -h

    usage: data_explorer.py [-h] [--vocab VOCAB] [--port PORT] [--disable-caching-metrics] [--estimate-audio-metrics] [--debug] manifest

    Speech Data Explorer

    positional arguments:
    manifest              path to JSON manifest file

    optional arguments:
    -h, --help            show this help message and exit
    --vocab VOCAB         optional vocabulary to highlight OOV words
    --port PORT           serving port for establishing connection
    --disable-caching-metrics
                            disable caching metrics for errors analysis
    --estimate-audio-metrics, -a
                            estimate frequency bandwidth and signal level of audio recordings
    --debug, -d           enable debug mode
    --audio-base-path A base path for the relative paths in manifest. It defaults to manifest path.
    --names_compared, -nc names of the two fields that will be compared, example: pred_text_contextnet pred_text_conformer.
    --show_statistics, -shst field name for which you want to see statistics (optional). Example: pred_text_contextnet.

Comparison tool (in SDE) takes as an input a JSON manifest file (that describes speech datasets in NeMo). It should contain the following fields:

* `audio_filepath` (path to audio file)
* `duration` (duration of the audio file in seconds)
* `text` (reference transcript)
* `pred_text_<model_1_name>`
* `pred_text_<model_2_name>`

SDE supports any extra custom fields in the JSON manifest. If the field is numeric, then SDE can visualize its distribution across utterances.

JSON manifest has attribute `pred_text`, SDE interprets it as a predicted ASR transcript and computes error analysis metrics.
If you want SDE to analyse another prediction field - use --show_statistics argument.


User Interface
--------------

SDE application has tree pages if --names_compared argument is not empy:

* `Statistics` (to display global statistics and aggregated error metrics)

    .. image:: images/sde_base_stats.png
        :align: center
        :alt: SDE Statistics
        :scale: 100%

* `Samples` (to allow navigation across the entire dataset and exploration of individual utterances)

    .. image:: images/sde_player.png
        :align: center
        :alt: SDE Statistics
        :scale: 100%

* `Comparison tool` (To visually explore predictions)

    .. image:: images/scrsh_2.png
        :align: center
        :alt: Comparison tool
        :scale: 100%


Plotly Dash Datatable provides core CT's interactive features (navigation, filtering, and sorting).
Comparison tool has one datatable:


* Data (that visualizes all dataset's words and adds each one's accuracy)

    .. image:: images/scrsh_3.png
        :align: center
        :alt: Data
        :scale: 100%

CT supports all operations, that present in SDE and in addition "or" and "and" operations

* filtering (by entering a filtering expression in a cell below the header's cell): CT supports all operations, that present in SDE, and, in addition, "or" and "and" operations

    .. image:: images/scrsh_4.png
        :align: center
        :alt: Filtering
        :scale: 100%


Analysis of Speech Datasets
---------------------------

If there is a pre-trained ASR model, then the JSON manifest file can be extended with ASR predicted transcripts:

.. code-block:: bash

    python examples/asr/transcribe_speech.py pretrained_name=<ASR_MODEL_NAME> dataset_manifest=<JSON_FILENAME> append_pred=False pred_name_postfix=<model_name_1>
    

More information about transcribe_speech cold be found inside it's code: `NeMo/examples/asr/transcribe_speech.py <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py>`__.
.

    .. image:: images/scrsh_2.png
        :align: center
        :alt: fields
        :scale: 100%

Fields 1 and 2 are responsible for what will be displayed on the horizontal and vertical axes

Fields 3 and 4 allow you to convert any available numeric parameter into color and size, respectively.

Fields 5 and 6 are responsible for point spacing. With a high probability, some points will have the same coordinates on both axes, in which case there will be an overlap, and in order to be able to explore each point, the possibility of their separation was added

.

    .. image:: images/scrsh_5.png
        :align: center
        :alt: dot spacing
        :scale: 100%

Point spacing works as follows: a small random value is added to all point coordinates, the value of which is limited by the "radius" parameter, which can be set manually

.

    .. image:: images/scrsh_9.png
        :align: center
        :alt: Example
        :scale: 100%

Initially, the accuracy on the word of the first model is displayed along the axes, from the second, in this case, all points lying above the diagonal will have a higher accuracy on the model displayed on the vertical axis,
and all points below the diagonal will have a better quality on the model displayed on the horizontal axis

Points marked with circles should be explored first

Words in the first quarter were well recognized by both models, and conversely, words in the third quarter were poorly recognized by both models.