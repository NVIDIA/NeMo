8kHz Models
===========

For applications based on 8 kHz speech, we recommend using the Quartznet model available `here <https://ngc.nvidia.com/catalog/models/nvidia:multidataset_quartznet15x5>`__.
This model for 16 kHz speech was trained on multiple datasets including about 1700 hours of upsampled narrowband conversational telephone speech from
Fisher and Switchboard datasets. In our experiments, we found that this model's accuracy on 8 kHz
speech is at par with a model trained exclusively on narrowband speech.

For best accuracy in your application, you may choose to :ref:`fine-tune <fine-tune>` this model using data collected from your application.


