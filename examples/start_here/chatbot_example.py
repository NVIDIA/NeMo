import gzip
import os
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
    # "num_epochs": 15,
    # 3 is too small - used for test
    "num_epochs": 3,
    "optimizer_kind": "adam",
    "learning_rate": 0.0003,
    "tb_log_dir": "ChatBot",
}

# instantiate neural factory
nf = nemo.core.NeuralModuleFactory()
# To use CPU-only do:
# from nemo.core import DeviceType
# nf = nemo.core.NeuralModuleFactory(placement=DeviceType.CPU)

# instantiate neural modules
dl = nemo.tutorials.DialogDataLayer(**config)
encoder = nemo.tutorials.EncoderRNN(**config)
decoder = nemo.tutorials.LuongAttnDecoderRNN(**config)
L = nemo.tutorials.MaskedXEntropyLoss()
decoderInfer = nemo.tutorials.GreedyLuongAttnDecoderRNN(**config)

# PARAMETER SHARING: between training and auto-regressive inference decoders
decoderInfer.tie_weights_with(decoder, list(decoder.get_weights().keys()))

# express activations flow
src, src_lengths, tgt, mask, max_tgt_length = dl()
encoder_outputs, encoder_hidden = encoder(input_seq=src, input_lengths=src_lengths)
outputs, hidden = decoder(targets=tgt, encoder_outputs=encoder_outputs, max_target_len=max_tgt_length)
loss = L(predictions=outputs, target=tgt, mask=mask)

# run inference decoder to generate predictions
outputs_inf, _ = decoderInfer(encoder_outputs=encoder_outputs)


# define callback function which prints intermediate results to console
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
    print(f"Train Loss:{str(tensors[0].item())}")
    print(f"SOURCE: {source} <---> PREDICTED RESPONSE: {response} " f"<---> TARGET: {target}")


callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[loss, src, outputs_inf, tgt], print_func=lambda x: outputs2words(x, dl.voc.index2word)
)

# start training
nf.train(
    tensors_to_optimize=[loss],
    callbacks=[callback],
    optimizer="adam",
    optimization_params={"num_epochs": config["num_epochs"], "lr": 0.001},
)
