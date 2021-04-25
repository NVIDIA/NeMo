Inverse Text Normalization Deployment
=====================================

This tool deploys :doc:`NeMo Inverse Text Normalization <../nemo_text_processing/inverse_text_normalization>` for production.
It uses Sparrowhawk - an open-source version of Google Kestrel :cite:`tools-itn_deploy-ebden2015kestrel`.

Requirements
------------

Install the :doc:`nemo_text_processing <../nemo_text_processing/intro>` package.

Usage
-----

#. Start the Docker container with production backend, with plugged-in grammars. This is an entry point script.

   .. code-block:: bash

       bash export_grammar.sh

   This script runs the following steps in sequence:

#. Export grammars ``tokenize_and_classify_tmp.far`` and ``verbalize_tmp.far`` from :doc:`nemo_text_processing <../nemo_text_processing/intro>` 
   to directory ``classify/`` and ``verbalize/`` respectively.

   .. code-block:: bash

       python pynini_export.py OUTPUT_DIR

   The script uses the output of this step to compile the final grammars ``tokenize_and_classify.far`` and ``verbalize.far`` for 
   deployment. 

   .. code-block:: bash

       cd classify; thraxmakedep tokenize_and_classify.grm ; make; cd ..
       cd verbalize; thraxmakedep verbalize.grm ; make; cd ..

#. BuildsC++ production backend Docker.

   .. code-block:: bash

       bash docker/build.sh

#. Plug in grammars into production backend by mounting the grammar directory ``classify/`` and ``verbalize/`` with sparrowhawk 
   grammar directory inside Docker. Returns a Docker prompt:

   .. code-block:: bash

       bash docker/launch.sh

#. Run the Inverse Text Normalization: 

   .. code-block:: bash

       echo "two dollars fifty" | ../../src/bin/normalizer_main --config=sparrowhawk_configuration.ascii_proto

   This returns $2.50.

References
----------

.. bibliography:: tools_all.bib
    :style: plain
    :labelprefix: TOOLS-ITN_DEPLOY
    :keyprefix: tools-itn_deploy-
