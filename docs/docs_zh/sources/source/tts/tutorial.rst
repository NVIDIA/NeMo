教程
========

请确保您已经安装了 ``nemo``，``nemo_asr``，和 ``nemo_tts``
模块。参考 :ref:`installation` 章节。

.. note::
    本教程仅要求安装 `nemo`，`nemo_asr`，和 `nemo_tts` 模块

介绍
-------------
语音合成，又被称为文本转语音(TTS)，通常指根据文本合成人类说话的声音。使用神经网络进行
语音合成通常包含两个神经网络模型。第一个模型能够将文本转化为中间态的音频表征，通常为音频谱(spectrogram)。
第二个模型称为声码器，能够将中间态的音频表征转化为声音文件，一种我们常见的声音文件格式为 .wav。
尽管目前有一些研究表示可以将这两个模型合并为一个独立的模型，在本教程中，我们关注于使用两个模型的方法。

NeMo 支持以下两个模型：

1. `Tacotron 2 <https://arxiv.org/abs/1712.05884>`_ ：该模型用于将文本转化为梅尔频谱
2. `Waveglow <https://arxiv.org/abs/1811.00002>`_ ： 该模型用于将梅尔频谱转化为声音文件

要想使用 NeMo 训练语音合成模型，你可以继续阅读以下章节。如果你想使用预先训练好的模型直接合成语音，
请跳转到 :ref:`语音合成 <语音合成>` 章节。

获取数据
--------
Tacotron 2 和 Waveglow 都可以使用
`LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`__ 数据集来训练。
你可以使用一个辅助脚本来获得用于 NeMo 训练的数据，该脚本位于 NeMo/scripts，请按照如下方式运行该脚本：

.. code-block:: bash

    python scripts/get_ljspeech_data.py --data_root=<where_you_want_to_save_data>

想了解更多关于 LJSpeech 数据集的细节，可以参考 :ref:`这里 <ljspeech_dataset>`。

对于普通话语音合成的数据:`中文标准女声音库 <https://www.data-baker.com/open_source.html>`__，你也可以使用一个辅助脚本来获得，该脚本位于 NeMo/scripts，
辅助脚本中使用的数据集下载链接由标贝（北京）科技有限公司提供。

.. code-block:: bash

    python scripts/get_databaker_data.py --data_root=<where_you_want_to_save_data>

想了解更多关于中文标准女声音库数据集的细节，可以参考 :ref:`这里 <中文标准女声音库>`。

训练
---------
NeMo 支持对 Tacotron 2 和 Waveglow 进行训练。在本教程中，我们主要关注于 Tacotron 2 模型的训练， 
因为其负责从训练数据的音频中学习大部分的特征，例如性别、韵律等。 另外，我们的实验还表明，
Waveglow 作为声码器是具有通用型的。具体的表现为，我们在一个英文女声语音数据集上训练得到的 
Waveglow 模型，可以用于男性声音或者其他语言（如普通话）的声码器。

训练 Tacotron 2 可以通过运行 `tacotron2.py` 文件来完成，该脚本位于 
NeMo/examples/tts。假设你当前已经位于 NeMo/examples/tts 目录下，
你可以通过运行如下命令开始训练：

.. code-block:: bash

    python tacotron2.py --train_dataset=<data_root>/ljspeech_train.json --eval_datasets <data_root>/ljspeech_eval.json --model_config=configs/tacotron.yaml --max_steps=30000

使用普通话数据进行训练也可以通过运行 `tacotron2.py` 文件来完成，
你可以通过运行如下命令开始训练：

.. code-block:: bash

    python tacotron2.py --train_dataset=<data_root>/databaker_csmsc_train.json --eval_datasets <data_root>/databaker_csmsc_eval.json --model_config=configs/tacotron_mandarin.yaml --max_steps=30000
    
.. tip::
    Tacotron 2 通常需要约 20,000 个训练步来学习到正确的注意力（也可以理解为对齐）。
    一旦模型学习到了正确的注意力，你就可以使用该模型来合成较为清晰的语音。

混合精度训练
-------------------------
启用或关闭混合精度训练可以通过一个命令行参数来控制 --amp_opt_level。对于 Tacotron 2
和 Waveglow 来说，该参数建议的默认值为 O1。该参数值可以设置为以下几种：

- O0: 单精度（float32）训练
- O1: 混合精度训练
- O2: 混合精度训练
- O3: 半精度（float16）训练

.. note::
    混合精度依赖 Tensor Cores ，NVIDIA 的 Volta 和 Turing 架构 GPU 支持 Tensor Cores。

多 GPU 训练
-------------------
要想启用在多个 GPU 上训练可以通过在运行训练脚本时调用
torch.distributed.launch 模块并指定 --nproc_per_node 参数为 GPU 的数量：

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=<num_gpus> <nemo_git_repo_root>/examples/tts/tacotron2.py ...


.. _语音合成:

合成语音
---------
你可以使用自己训练的 Tacotron 2 模型合成语音，也可以使用我们预训练好的 Tacotron 2 模型合成语音。 
下一步，请创建你想用于语音合成的文本，并将其转化为训练数据格式相同的 JSON 格式。该 JSON 文件格式如下所示：

.. code-block:: json

    {"audio_filepath": "", "duration": 1.0, "text": "Talk to me!"}
    {"audio_filepath": "", "duration": 1.0, "text": "Speech Synthensis is cool."}

如果要合成普通话语音，JSON 文件格式如下所示：

.. code-block:: json

    {"audio_filepath": "", "duration": 1.0, "text": "jin1 tian1 tian1 qi4 bu2 cuo4."}
    {"audio_filepath": "", "duration": 1.0, "text": "ni3 kan4 bao4 zhi3 ma0"}

其中 “text” 字段包含想要合成的语音的拼音序列，每个拼音后的数字（0-4）代表该发音的声调，0 代表轻声。

语音合成可以通过运行 NeMo/examples/tts 文件夹下的 tts_infer.py 脚本完成，你可以通过如下命令运行该脚本：

.. code-block:: bash

    python tts_infer.py --spec_model=tacotron2 --spec_model_config=configs/tacotron2.yaml --spec_model_load_dir=<directory_with_tacotron2_checkopints> --vocoder=waveglow --vocoder_model_config=configs/waveglow.yaml --vocoder_model_load_dir=<directory_with_waveglow_checkopints> --save_dir=<where_you_want_to_save_wav_files> --eval_dataset <mainfest_to_generate>

要合成普通话语音，记得将 Tacotron 2 模型配置文件更换为 tacotron2_mandarin.yaml。
