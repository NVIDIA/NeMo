.. _tacotron-2:

Tacotron 2
==========

Model
~~~~~
本模型基于 `Tacotron 2 模型 <https://ai.googleblog.com/2017/12/tacotron-2-generating-human-like-speech.html>`_
(也可参考 `文献 <https://arxiv.org/abs/1712.05884>`_)。

Tacotron 2 模型使用的编码器-解码器结构能够很好地对序列到序列（sequence-to-sequence）建模，NeMo 将 Tacotron 2 分成了 4 个不同的神经模块:

1. 文本嵌入（TextEmbedding）神经模块包含的查找表用于将字符 id 转化为嵌入空间中的向量表示。
2. 嵌入表示的文本序列会传入到 Tacotron2Encoder 神经模块中。
3. Tacotron2Decoder 神经模块包含模型中的注意力机制以及解码器的循环神经网络（RNN）部分。
4. 最后, Tacotron2Postnet 神经模块对 Tacotron2Decoder 神经模块中编码器输出的音频谱进行修正。

小建议
~~~~~~~
模型通过注意力机制学习到的对齐矩阵可以用来初步衡量合成音频的质量。在理想情况下，我们希望
学习到的对齐是一个干净清晰的对角线。NeMo 中当前的模型能够在训练 20k 个训练步后学习到输入
文本和输出音频谱的对齐信息。
