Waveglow
========

`Waveglow <https://arxiv.org/abs/1811.00002>`_ is an universal neural vocoder
based on normalizing flows to generate audio. It converts mel spectrograms
to waveforms.

NeMo currently does not decompose WaveGlow down to separate Neural Modules but
rather treats the entire model as a Neural Module.

Tips
~~~~
Our pre-trained Waveglow should be able to used as a vocoder for most if not
all languages! One can always train Waveglow on their own data by running
waveglow.py from NeMo/examples/tts like so:

.. code-block:: bash

    python waveglow.py --train_dataset=<data_root>/ljspeech_train.json --eval_datasets <data_root>/ljspeech_eval.json --model_config=configs/waveglow.yaml --num_epochs=1500

Please note that training waveglow will take longer than training tacotron 2.
