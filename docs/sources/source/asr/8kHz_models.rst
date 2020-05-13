8kHz Models
===========

For applications based telephony speech, using models trained on narrowband audio data sampled at 8 kHz may perform better than using models built with
audio at a higher frequency (Note that to use models with audio at a different sample rate from your data, you would need to resample your data to match the sampling rate in the
config file of the model). One approach to create large datasets for training a model suitable for your application would be to convert all audio data
to the formats prevalent in your application. Here we detail one such approach that we took to train a model based on 8 kHz data.

To train a model suitable for recognizing telephony speech we converted some of the datasets to G.711 :cite:`8kHz-mod-itu1988g711`. G.711 is a popular speech codec used in VoIP products and encodes speech
at 64 kbps using PCM u-law companding. We converted audio from LibriSpeech, Mozilla Common Voice and WSJ datasets to G.711 format and combined Fisher and Switchboard datasets to
train a Quartznet15x5 model with about 4000 hours of data. The model was first pretrained with 8 kHz LibriSpeech data for 134 epochs and then trained for another 250 epochs using G.711 audio from all the five datasets listed above. For best accuracy
in your application, you may choose to :ref:`fine-tune <fine-tune>` this model using data collected from your application. The pre-trained model is available
for download `here <https://ngc.nvidia.com/catalog/models/nvidia:quartznet15x5>`.

References
----------
.. bibliography:: asr_all.bib
   :style: plain
   :labelprefix: 8kHz-mod
   :keyprefix: 8kHz-mod-
