# Copyright (c) 2019 NVIDIA Corporation
import gzip
import os
import shutil

import nemo
from nemo import logging

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

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch, local_rank=None)

# instantiate necessary neural modules
dl = neural_factory.get_module(name="DialogDataLayer", collection="tutorials", params=config)

# Instance one on EncoderRNN
encoder1 = neural_factory.get_module(name="EncoderRNN", collection="tutorials", params=config)
# Instance two on EncoderRNN. It will have different weights from instance one
encoder2 = neural_factory.get_module(name="EncoderRNN", collection="tutorials", params=config)
mixer = neural_factory.get_module(name="SimpleCombiner", collection="common", params={})

decoder = neural_factory.get_module(name="LuongAttnDecoderRNN", collection="tutorials", params=config)

L = neural_factory.get_module(name="MaskedXEntropyLoss", collection="tutorials", params={})

decoderInfer = neural_factory.get_module(name="GreedyLuongAttnDecoderRNN", collection="tutorials", params=config)
# notice trainng and inference decoder share parameters
decoderInfer.tie_weights_with(decoder, list(decoder.get_weights().keys()))

# express activations flow
src, src_lengths, tgt, mask, max_tgt_length = dl()
encoder_outputs1, encoder_hidden1 = encoder1(input_seq=src, input_lengths=src_lengths)
encoder_outputs2, encoder_hidden2 = encoder2(input_seq=src, input_lengths=src_lengths)
encoder_outputs = mixer(x1=encoder_outputs1, x2=encoder_outputs2)
outputs, hidden = decoder(targets=tgt, encoder_outputs=encoder_outputs, max_target_len=max_tgt_length)
loss = L(predictions=outputs, target=tgt, mask=mask)

# run inference decoder to generate predictions
outputs_inf, _ = decoderInfer(encoder_outputs=encoder_outputs)


# this function is necessary to print intermediate results to console


def outputs2words(tensors, vocab):
    source_ids = tensors[1][:, 0].cpu().numpy().tolist()
    response_ids = tensors[2][:, 0].cpu().numpy().tolist()
    tgt_ids = tensors[3][:, 0].cpu().numpy().tolist()
    source = list(map(lambda x: vocab[x], source_ids))
    response = list(map(lambda x: vocab[x], response_ids))
    target = list(map(lambda x: vocab[x], tgt_ids))
    source = ' '.join([s for s in source if s != 'EOS' and s != 'PAD'])
    response = ' '.join([s for s in response if s != 'EOS' and s != 'PAD'])
    target = ' '.join([s for s in target if s != 'EOS' and s != 'PAD'])
    logging.info(f'Train Loss: {str(tensors[0].item())}')
    tmp = " SOURCE: {0} <---> PREDICTED RESPONSE: {1} <---> TARGET: {2}"
    return tmp.format(source, response, target)


# Create trainer and execute training action
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[loss, src, outputs_inf, tgt], print_func=lambda x: outputs2words(x, dl.voc.index2word),
)
# Instantiate an optimizer to perform `train` action
optimizer = neural_factory.get_trainer()

optimizer.train(
    tensors_to_optimize=[loss],
    callbacks=[callback],
    optimizer="adam",
    optimization_params={"num_epochs": config["num_epochs"], "lr": 0.001},
)
