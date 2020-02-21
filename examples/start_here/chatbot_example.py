import gzip
import os
import shutil

import nemo

logging = nemo.logging

data_file = "movie_data.txt"

# Download the data file.
if not os.path.isfile(data_file):
    with gzip.open("../../tests/data/movie_lines.txt.gz", 'rb') as f_in:
        with open(data_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


# Instantiate the neural factory
nf = nemo.core.NeuralModuleFactory()
# To use CPU-only do:
# nf = nemo.core.NeuralModuleFactory(placement=nemo.core.DeviceType.CPU)

# Instantiate all required neural modules.
dl = nemo.tutorials.DialogDataLayer(batch_size=128, corpus_name="cornell", datafile=data_file)
encoder = nemo.tutorials.EncoderRNN(voc_size=(6104 + 3), encoder_n_layers=2, hidden_size=512, dropout=0.1)
decoder = nemo.tutorials.LuongAttnDecoderRNN(
    attn_model="dot", hidden_size=512, voc_size=(6104 + 3), decoder_n_layers=2, dropout=0.1
)
L = nemo.tutorials.MaskedXEntropyLoss()

decoderInfer = nemo.tutorials.GreedyLuongAttnDecoderRNN(
    attn_model="dot", hidden_size=512, voc_size=(6104 + 3), decoder_n_layers=2, dropout=0.1, max_dec_steps=10
)

# PARAMETER SHARING: between training and auto-regressive inference decoders.
decoderInfer.tie_weights_with(decoder, list(decoder.get_weights().keys()))

# Connect the modules - express activations flow for training.
src, src_lengths, tgt, mask, max_tgt_length = dl()
encoder_outputs, encoder_hidden = encoder(input_seq=src, input_lengths=src_lengths)
outputs, hidden = decoder(targets=tgt, encoder_outputs=encoder_outputs, max_target_len=max_tgt_length)
loss = L(predictions=outputs, target=tgt, mask=mask)

# Run inference decoder to generate predictions.
outputs_inf, _ = decoderInfer(encoder_outputs=encoder_outputs)


# Define the callback function which prints intermediate results to console.
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
    logging.info(f"Train Loss:{str(tensors[0].item())}")
    logging.info(f"SOURCE: {source} <---> PREDICTED RESPONSE: {response} " f"<---> TARGET: {target}")


# Create simple callback.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[loss, src, outputs_inf, tgt], print_func=lambda x: outputs2words(x, dl.voc.index2word),
)

# num_epochs = 1
max_steps = 50
# logging.info(f"Training only for {num_epochs}. Train longer (~10-20) for convergence.")
logging.info(f"Training only for {max_steps} steps. Train longer (~10-20) for convergence.")
# Start training
nf.train(
    tensors_to_optimize=[loss],
    callbacks=[callback],
    optimizer="adam",
    optimization_params={"max_steps": max_steps, "lr": 0.001},
)
