Examples
========

Nemo applications consist of 3  stages:

    (1) create NeuralModules 
    (2) construct model using tensors ('activations') to connect modules into graph
    (3) start action, e.g. training or inference


Hello World 
------------

This example shows how to build a model which learn Taylor's coefficients for y=sin(x).


.. code-block:: python

    import nemo

    # instantiate Neural Factory 
    nf = nemo.core.NeuralModuleFactory()

    # instantiate neural modules
    data = nf.get_module(name="RealFunctionDataLayer", collection="toys",
                         params={"n": 10000, "batch_size": 128})
    f = nf.get_module(name="TaylorNet", collection="toys", params={"dim": 4})
    L = nf.get_module(name="MSELoss", collection="toys", params={})

    # build model out of neural modules using activations 
    x, y = data()
    p = f(x=x)
    loss = L(predictions=p, target=y)

    # add SimpleLossLoggerCallback to print loss values to console
    # Callback function converts a list of tensors into string
    callback = nemo.core.SimpleLossLoggerCallback(tensor_list2string=lambda x: str(x[0].item()))
    
    # instantiate SGD as optimizer  
    optimizer = nf.get_trainer(params={"optimization_params": {"num_epochs": 3, "lr": 0.0003}})
    
    # start training
    optimizer.train([loss], callbacks=[callback])

Simple Chatbot
---------------

This is an adaptation of `PyTorch's Chatbot tutorial <https://pytorch.org/tutorials/beginner/chatbot_tutorial.html>`_ into NeuralModule's framework. It demonstrates how to do training and evaluation.

Model can be describes by graph shown below. Model has:
 
	* two data layers (one for training and another one for inference), 
	* encoder and decoder (shared by training and inference), 
	* two loss modules (one for training and another one for inference). 

.. image:: chatbot.png

During training model will print:

	* **SOURCE**:  model input
	* **PREDICTED RESPONSE**: model output
	* **TARGET**:  target output 

.. code-block:: python

    import os
    import sys
    import gzip
    import shutil
    import nemo

    # Get Data
    data_file = "movie_data.txt"
    if not os.path.isfile(data_file):
      with gzip.open("../../tests/data/movie_lines.txt.gz", 'rb') as f_in:
        with open(data_file, 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)

    # Configuration
    config = {
      "corpus_name": "cornell",
      "datafile": data_file,
      "attn_model": 'dot',
      "hidden_size": 512,
      "encoder_n_layers": 2,
      "decoder_n_layers": 2,
      "dropout": 0.1,
      "voc_size": 6104 + 3,
      "batch_size": 128,
      "num_epochs": 15,
      "optimizer_kind": "adam",
      "learning_rate": 0.0003,
      "tb_log_dir": "ChatBot",
    }

    #instantiate neural factory 
    nf = nemo.core.NeuralModuleFactory()

    #instantiate neural modules
    dl = nf.get_module(name="DialogDataLayer", collection="tutorials", params=config)

    encoder = nf.get_module(name="EncoderRNN", collection="tutorials", params=config)

    decoder = nf.get_module(name="LuongAttnDecoderRNN", collection="tutorials", params=config)

    L = nf.get_module(name="MaskedXEntropyLoss", collection="tutorials", params={})

    decoderInfer = nf.get_module(name="GreedyLuongAttnDecoderRNN", collection="tutorials", params=config)

    # PARAMETER SHARING: between training and auto-regressive inference decoders
    decoderInfer.tie_weights_with(decoder, list(decoder.get_weights().keys()))

    # express activations flow
    src, src_lengths, tgt, mask, max_tgt_length = dl()
    encoder_outputs, encoder_hidden = encoder(input_seq=src, input_lengths=src_lengths)
    outputs, hidden = decoder(targets=tgt, encoder_outputs=encoder_outputs,
                              max_target_len=max_tgt_length)
    loss = L(predictions=outputs, target=tgt, mask=mask)

    # run inference decoder to generate predictions
    outputs_inf, _ = decoderInfer(encoder_outputs=encoder_outputs)

    # define callback function which prints intermediate results to console
    def outputs2words(tensors, vocab):
      source_ids = tensors[0][:, 0].cpu().numpy().tolist()
      response_ids = tensors[1][:, 0].cpu().numpy().tolist()
      tgt_ids = tensors[2][:, 0].cpu().numpy().tolist()
      source = list(map(lambda x: vocab[x], source_ids))
      response = list(map(lambda x: vocab[x], response_ids))
      target = list(map(lambda x: vocab[x], tgt_ids))
      source = ' '.join([s for s in source if s!='EOS' and s!='PAD'])
      response = ' '.join([s for s in response if s!='EOS' and s!='PAD'])
      target = ' '.join([s for s in target if s!='EOS' and s!='PAD'])
      return " SOURCE: {0} <---> PREDICTED RESPONSE: {1} <---> TARGET: {2}".format(
        source, response, target)

    callback = nemo.core.SimpleLossLoggerCallback(
      tensor_list2string=lambda x: str(x[0].item()),
      tensor_list2string_evl=lambda x: outputs2words(x, dl.voc.index2word))
    
    # instantiate an optimizer for training 
    optimizer = nf.get_trainer(params={"optimizer_kind": "adam",
                  "optimization_params": {"num_epochs": config["num_epochs"], "lr": 0.001}})

    # start training
    optimizer.train(tensors_to_optimize=[loss], tensors_to_evaluate=[src, outputs_inf, tgt],
                    callbacks=[callback])

.. note::
    Look for more examples under `nemo/examples`

