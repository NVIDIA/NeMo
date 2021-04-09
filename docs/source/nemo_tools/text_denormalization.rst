Inverse Text Normalization
==========================

Inverse text normalization (ITN), also called denormalization, is a part of the Automatic Speech Recognition (ASR) post-processing pipeline.
ITN is the task of converting the raw spoken output of the ASR model into its written form to improve text readability.

For example, 
`"in nineteen seventy"` -> `"in 1975"` 
and `"one hundred and twenty three dollars"` -> `"$123"`.

This tool is based on WFST-grammars :cite:`tools-denorm-mohri2009`. We also provide a deployment route to C++ using Sparrowhawk -- an open-source version of Google Kestrel :cite:`tools-denorm-ebden2015kestrel`.
See :doc:`tools/inverse_text_denormalization <../tools/intro>` for details.

.. note::

    For more details, see the tutorial `NeMo/tutorials/tools/Text_Denormalization_(Inverse_Text_Normalization).ipynb <https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/Text_Denormalization_(Inverse_Text_Normalization).ipynb>`__.




References
----------

.. bibliography:: tools_all.bib
    :style: plain
    :labelprefix: TOOLS-DENORM
    :keyprefix: tools-denorm-