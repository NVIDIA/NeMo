Weight Sharing between Modules
==============================

There are several ways to share or tie weights between neural models.

Neural Module reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~

The idea is to re-use neural modules between training, evaluation and inference graphs.
For example:

.. code-block:: python

    ...
    train_dataloader=nf.get_module(name="TrainDataLayer", params=train_config)
    eval_dataloader = nf.get_module(name="EvalDataLayer", params=eval_config)

    L = nf.get_module(name="MaskedXEntropyLoss", params={})

    # training model

    src, src_lengths, tgt, mask, max_tgt_length = train_dataloader()
    encoder_outputs, encoder_hidden = encoder(input_seq=src, input_lengths=src_lengths)
    outputs, hidden = decoder(targets=tgt, encoder_outputs=encoder_outputs, 
                              max_target_len=max_tgt_length)
    train_loss = L(predictions=outputs, target=tgt, mask=mask)


    # evaluation model

    src, src_lengths, tgt, mask, max_tgt_length = eval_dataloader()
    encoder_outputs, encoder_hidden = encoder(input_seq=src, input_lengths=src_lengths)
    outputs, hidden = decoder(targets=tgt, encoder_outputs=encoder_outputs, 
                              max_target_len=max_tgt_length)
    eval_loss = L(predictions=outputs, target=tgt, mask=mask)
    ...


Copy weights between modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` class provides 2 methods
:meth:`get_weights<nemo.core.neural_modules.NeuralModule.get_weights>` and
:meth:`set_weights<nemo.core.neural_modules.NeuralModule.set_weights>` 
for sharing weights.

.. note::
    :meth:`set_weights<nemo.core.neural_modules.NeuralModule.set_weights>` method can set only part of module's weights.

.. important::
    This approach is used only to copy weights. Subsequent update of weights in one module will not affect weights in the other module.

Consider an example:

.. code-block:: python

    tn1 = nemo.pytorch.toys.TaylorNet(dim=4)
    tn2 = nemo.pytorch.toys.TaylorNet(dim=4)

    # because of random intialization, weights should be different
    self.assertFalse(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                       tn2.get_weights()))
    tn3 = nemo.pytorch.toys.TaylorNet(dim=4)
    tn3.set_weights(tn1.get_weights())

    # check than weights are the same
    self.assertTrue(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                      tn3.get_weights()))

    # change weights in tn1 module - another module should not change
    tn1.fc1.bias.data = torch.tensor([0.1])
    self.assertFalse(self.__check_if_weights_are_equal(tn1.get_weights(),
                                                       tn3.get_weights()))


Tie weights between modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` class provides :meth:`tie_weights_with<nemo.core.neural_modules.NeuralModule.tie_weights_with>` method to tie weights between two or more modules.

.. important::
    Tied weights are identical across all modules: subsequent modification of weights in one module will result in the same modification on the other.


In the example below we first create a simple embedding encoder which takes [batch, time] sequences of word ids from vocabulary ``V``  and embeds them into some ``D``-dimensional space. Effectively, this is a lookup-based projection from ``V``-dimensional space to ``D``-dimensional space. We then create a decoder which projects from ``D``-dimensional space back to the ``V``-dimensional space. We want to transpose the encoder projection matrix and reuse it for decoder.
The code below demonstrates how this can be achieved.

.. note::
   The weights have different names (``embedding.weight`` and ``projection.weight``) but their values are the same. Changes to one will result to changes in the other. Effectively, ``embedding.weight`` and ``projection.weight`` become pointers to the same tensor.


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

    was = embd.embedding.weight.detach().numpy()

    # Now, change weights on one object
    embd.embedding.weight.data = torch.tensor(np.random.randint(0, 10, (3, 2))*1.0)
    after = embd.embedding.weight.detach().numpy()

    # Make sure that the change was reflected on another object
    self.assertTrue(np.array_equal(embd.embedding.weight.detach().numpy(),
                                    proj.projection.weight.detach().numpy()))
    self.assertFalse(np.array_equal(was, after))

