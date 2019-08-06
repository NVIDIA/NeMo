NEMO: NEural MOdules Toolkit
============================

.. image:: docs/sources/source/nemo-icon-256x256.png
    :width: 40
    :align: center
    :alt: NEMO

NEural MOdules (NEMO): a framework-agnostic toolkit for building AI applications powered by Neural Modules


**VIDEO**

`A Short VIDEO walk-through about using NEMO to experiment with ASR systems. <https://confluence.nvidia.com/display/AAD/Neural+Modules?preview=/163999167/299604998/nemo_for_asr.mp4>`_


.. raw:: html

    <figure class="video_container">
      <video controls="true" allowfullscreen="true">
        <source src="https://confluence.nvidia.com/display/AAD/Neural+Modules?preview=/163999167/299604998/nemo_for_asr.mp4" type="video/mp4">
      </video>
    </figure>
    <!-- blank line -->



**Core Concepts and Features**

* `NeuralModule` class - represents and implements a neural module.
* `NmTensor` - represents activation flow between neural modules' ports.
* `NeuralType` - represents types of modules' ports and NmTensors.
* `NeuralFactory` - to create neural modules and manage training.
* **Lazy execution** - when describing activation flow between neural modules, nothing happens until an "action" (such as `optimizer.optimize(...)` is called.
* **Collections** - NEMO comes with collections - related group of modules such as `nemo_asr` (for Speech Recognition) and `nemo_nlp` for NLP


**Requirements**

1) Python 3.6 or 3.7
2) Pytorch >=1.0 with GPU support
3) NVIDIA APEX: https://github.com/NVIDIA/apex
4) (for `nemo_asr` do: `apt-get install libsndfile1`)


**Unittests**

Run this:

.. code-block:: bash

    ./reinstall.sh
    python -m unittest tests/*.py


**Getting started**

1) Clone the repository and run unittests
2) Go to `nemo` folder and do: `python setup.py install`
3) Install collections:
    a) ASR collection from `collections/nemo_asr` do: `python setup.py install`
    b) NLP collection from `collections/nemo_nlp` do: `python setup.py install`
    c) LPR collection from `collections/nemo_lpr` do: `python setup.py install`
4) For development do: `python setup.py develop` instead of `python setup.py install` in Step (3) above
5) Go to `examples/start_here` to get started with few simple examples
6) To get started with speech recognition:

.. code-block:: bash

    cd examples/asr
    #download prepared AN4 dataset (an4data.tar.gz) from: https://drive.google.com/file/d/1n2CkS7KyTi5vb8qZm-HfSvrQcZVnmBuj
    tar -xvf an4data.tar.gz
    python jasper_an4.py

**Documentation**

(**WORK IN PROGRESS**)
http://10.110.40.155:8000
