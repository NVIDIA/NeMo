数据集
========

.. _LJSpeech:

LJSpeech
--------

`LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`_ 数据集中包含一个女性英语说话人的语音数据，总长度大约为 24 个小时。

在 NeMo 中获取并预处理该数据可以通过我们提供的
`辅助脚本 <https://github.com/NVIDIA/NeMo/blob/master/scripts/get_ljspeech_data.py>`_来完成:

.. code-block:: bash

    python scripts/get_ljspeech_data.py --data_root=<where_you_want_to_save_data>

.. _中文标准女声音库:

中文标准女声音库（10000句）
--------

`中文标准女声音库（10000句） <https://www.data-baker.com/open_source.html>`_ 数据集中包含一个女性普通话说话人的语音数据，总长度大约为 12 个小时。该数据集版权归标贝（北京）科技有限公司所有，仅支持非商用。

在 NeMo 中获取并预处理该数据可以通过我们提供的
`辅助脚本 <https://github.com/NVIDIA/NeMo/blob/master/scripts/get_databaker_data.py>`_来完成:

.. code-block:: bash

    python scripts/get_databaker_data.py --data_root=<where_you_want_to_save_data>