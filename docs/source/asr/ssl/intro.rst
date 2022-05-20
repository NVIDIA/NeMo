Self-Supervised Learning (SSL)
==================================

SSL, or Self-Supervised Learning, refers to the problem of learning without explicit labels. As 
any learning process require feedback, without explit labels, SSL derives supervisory signals from 
the data itself. The general ideal of SSL is to predict any hidden part (or property) of the input 
from observed part of the input (e.g., filling in the blanks in a sentence or predicting whether 
an image is upright or inverted).

SSL for speech/audio understanding broadly falls into either contrastive or reconstruction 
approaches. In contrastive approaches, models learn by distinguising between true and distractor 
tokens (or latents). Examples of contrastive approaches are Contrastive Predictive Coding (CPC), 
Masked Language Modeling (MLM) etc. In reconstruction approaches, models learn by directly estimating 
the missing (intentionally leftout) portions of the input. Masked Reconstruction, Autoregressive 
Predictive Coding (APC) are few examples. Here, we mainly focus on contrastive approaches. 

CTC based ASR neural models typically consist of an encoder and a decoder, where as a trasnducer based 
ASR model consists encoder, predictor and a joiner network. SSL for ASR typically involves pre-training 
the encoder network where as the other parts of the model are trained during down-stream task. However, 
the predictor part of transducer model can also be trained during SSL when applicable. 

The full documentation tree is as follows:

.. toctree::
   :maxdepth: 8

   models
   datasets
   results
   configs
   resources

.. include:: resources.rst
