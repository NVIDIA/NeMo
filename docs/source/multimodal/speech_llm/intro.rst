Speech-agumented Large Language Models (SpeechLLM)
==================================================

SpeechLLM is a multi-modal Large Language Model (LLM) designed to understand and process speech and audio inputs. Detailed information can be found in the `SpeechLLM examples README <https://github.com/NVIDIA/NeMo/blob/main/examples/multimodal/speech_llm/README.md>`_.

.. toctree::
   :maxdepth: 1

   datasets
   configs
   api


In general, there are three main components of a modular SpeechLLM:

- An audio encoder that processes the input audio and produces a sequence of audio embeddings.
- A modality adapter that processes the audio embeddings and produces a sequence of embeddings in the same latent space as the token embeddings of a pretrained LLM.
- A pretrained LLM that processes embeddings from the modality adapter and token embeddings from the input prompt, then produces the text output. The audio embeddings and text token embeddings are concatenated in time dimension before going into the LLM.
- The LLM produces text outputs based on the concatenated input audio and text embedding.


Model Architecture
^^^^^^^^^^^^^^^^^^

One way to incorporate speech into an LLM is to concatenate speech features with the token embeddings of the input text prompt before feeding them into the LLM. In this way, the LLM can have direct access to the speech information when generating the output text.

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.23.0/salm.png
    :align: center
    :alt: SALM model
    :scale: 50%



Another approach is to use a cross-attention mechanism, where text embeddings attend to speech embeddings to extract task-specific information. To minimize the computational cost, we add a cross-attention module only before the LLM.
   
.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.23.0/bestow.png
    :align: center
    :alt: BESTOW model
    :scale: 50%





