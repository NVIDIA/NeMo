模块间的权重共享
==============================

这里有一些在神经模块之间共享权重的方法。

神经模块重用
~~~~~~~~~~~~~~~~~~~~~~~~~~

这个想法是在训练，验证评估和推理图上重用神经模块。
例如:

.. code-block:: python

    ...
    train_dataloader = nemo.TrainDataLayer(**train_config)
    eval_dataloader = nemo.EvalDataLayer(**eval_config)

    L = nemo.tutorials.MaskedXEntropyLoss()

    # 训练模型

    src, src_lengths, tgt, mask, max_tgt_length = train_dataloader()
    encoder_outputs, encoder_hidden = encoder(input_seq=src, input_lengths=src_lengths)
    outputs, hidden = decoder(targets=tgt, encoder_outputs=encoder_outputs, 
                              max_target_len=max_tgt_length)
    train_loss = L(predictions=outputs, target=tgt, mask=mask)


    # 评测验证模型

    src, src_lengths, tgt, mask, max_tgt_length = eval_dataloader()
    encoder_outputs, encoder_hidden = encoder(input_seq=src, input_lengths=src_lengths)
    outputs, hidden = decoder(targets=tgt, encoder_outputs=encoder_outputs, 
                              max_target_len=max_tgt_length)
    eval_loss = L(predictions=outputs, target=tgt, mask=mask)
    ...


在模块间复制权重
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` 类提供了两个方法
:meth:`get_weights<nemo.core.neural_modules.NeuralModule.get_weights>` 和
:meth:`set_weights<nemo.core.neural_modules.NeuralModule.set_weights>` 
用来做权重复制

.. note::
    :meth:`set_weights<nemo.core.neural_modules.NeuralModule.set_weights>` 方法只能设置模块的部分权重

.. important::
    这个方法只能用来复制权重。后续在一个模块中更新权重不会影响到其他模块中的权重。
    这就意味着在更新步中，权重会得到不同的梯度。

考虑下面这个例子:

.. code-block:: python

    tn1 = nemo.pytorch.toys.TaylorNet(dim=4)
    tn2 = nemo.pytorch.toys.TaylorNet(dim=4)

    # 因为随机初始化，权重应该是不一样的
    self.assertFalse(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                       tn2.get_weights()))
    tn3 = nemo.pytorch.toys.TaylorNet(dim=4)
    tn3.set_weights(tn1.get_weights())

    # 检查权重是否一样
    self.assertTrue(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                      tn3.get_weights()))

    # 改变 tn1 模块中的权重 - 另一个模块中的权重不应该改变
    tn1.fc1.bias.data = torch.tensor([0.1])
    self.assertFalse(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                       tn3.get_weights()))


在模块间连接权重
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` 类提供 :meth:`tie_weights_with<nemo.core.neural_modules.NeuralModule.tie_weights_with>` 方法在多个模块间连接权重

.. important::
    连接后的权重在所有的模块之间保持一致，对于权重的梯度都是相同的。

.. important::
    然而手动通过 tensor.data 在一个模块上更新权重是不会更新 w 的。
    
在下面的例子中，我们首先创建一个简单的词嵌入编码器，它的输入是 [batch, time] 的词序列，从词表中 ``V`` 中找到词id，把它映射到 ``D`` 维空间。
这是一个查表的映射，从 ``V`` 维空间到 ``D`` 维空间。接着我们需要创建一个解码器，从 ``D`` 维空间映射到 ``V`` 维空间。我们想把编码器的映射矩阵在解码器中重用。下面的代码解释了这要怎么做。

.. note::
   权重有不同名字 (``embedding.weight`` 和 ``projection.weight``) 但值和梯度更新是一样的。


.. code-block:: python

    V = 3
    D = 2
    embd = nemo.pytorch.core.SequenceEmbedding(voc_size=V, hidden_size=D)
    proj = nemo.pytorch.core.SequenceProjection(from_dim=D, to_dim=voc_size)

    embd.tie_weights_with(proj, weight_names=["embedding.weight"],
                          name2name_and_transform={"embedding.weight":
                                                  ("projection.weight",
                                                   WeightShareTransform.SAME)})

    self.assertTrue(np.array_equal(embd.embedding.weight.detach().numpy(),
                                   proj.projection.weight.detach().numpy()))

.. warning::
    手动设置权重张量等于其它的张量有可能会打断多 GPU 和多节点(multi-node)运行，比如:
    ``embd.embedding.weight = proj.projection.weights`` 不太推荐，推荐使用 ``tie_weights_with()`` 函数


