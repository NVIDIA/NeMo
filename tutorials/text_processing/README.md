# NeMo Text Processing Tutorials

The NeMo Text Processing module provides support for both Text Normalization (TN) and 
Inverse Text Normalization (ITN) in order to aid upstream and downstream text processing.
The included tutorials are intended to help you quickly become familiar with the interface
of the module, as well as guiding you in creating and deploying your own grammars for individual
text processing needs.

If you wish to learn more about how to use NeMo's for Text Normalization tasks (e.g. conversion
of symbolic strings to verbal form - such as `15` -> "fifteen"), please see the [`Text Normalization`](https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_Normalization.ipynb)
tutorial.

If you wish to learn more about Inverse Text Normalization - the inverse task of converting 
from verbalized strings to symbolic written form, as may be encountered in downstream ASR - 
consult the [`Inverse Text Normalization`](https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Inverse_Text_Normalization.ipynb) tutorial.

For those curious about constructing grammars tailored to specific languages and use cases,
you may be interested in working through the [`WFST Tutorial`](https://github.com/NVIDIA/NeMo/blob/stable/tutorials/text_processing/WFST_Tutorial.ipynb), which goes through NeMo's Normalization
process in detail.

As NeMo Text Processing utilizes Weighted Finite State Transducer (WFST) graphs to construct its 
grammars, a working knowledge of [Finite State Automata](https://en.wikipedia.org/wiki/Finite-state_machine) (FSA) and/or regular languages is suggested.
Further, we recommend becoming functionally familiar with the [`pynini` library](https://www.openfst.org/twiki/bin/view/GRM/Pynini) - which functions
as the backend for graph construction - and [Sparrowhawk](https://github.com/google/sparrowhawk) - which NeMo utilizes for grammar deployment. 