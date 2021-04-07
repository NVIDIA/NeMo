**NeMo/tools/inverse_text_normalization**
=========================================

Introduction
------------

This folder provides scripts to deploy WFST-based grammars in `nemo_tools <https://github.com/NVIDIA/NeMo/blob/main/nemo_tools>`_ for
Inverse Text Normalization system in production.


Requirements
------------------------

1) nemo_tools

See NeMo `README <https://github.com/NVIDIA/NeMo/blob/main/README.rst>`_ for installation guide.


Usage
------------

Automatically start docker container with production backend with plugged in grammars. This is entry point script.

.. code-block:: bash

    bash export_grammar.sh

This scripts runs the following scripts in sequence:

Export grammars from nemo_tools

.. code-block:: bash

    python pynini_export.py


Build C++ production backend docker

.. code-block:: bash

    bash docker/build.sh


Plug in grammars into production backend and launch docker prompt

.. code-block:: bash

    bash docker/launch.sh


To run Inverse Text Normalization: 

.. code-block:: bash

    echo "two dollars fifty" | ../../src/bin/normalizer_main --config=sparrowhawk_configuration.ascii_proto

This should return $2.50.