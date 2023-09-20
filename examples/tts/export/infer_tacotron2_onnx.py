import soundfile as sf
import numpy as np
import onnxruntime as ort

# Load Tacotron2
from nemo.collections.tts.models import Tacotron2Model

# Load HifiGanModel
from nemo.collections.tts.models import HifiGanModel


def initialize_decoder_states(self, memory):
    B = memory.shape[0]
    MAX_TIME = memory.shape[1]

    attention_hidden = np.zeros((B, self.attention_rnn_dim), dtype=np.float32)
    attention_cell = np.zeros((B, self.attention_rnn_dim), dtype=np.float32)

    decoder_hidden = np.zeros((B, self.decoder_rnn_dim), dtype=np.float32)
    decoder_cell = np.zeros((B, self.decoder_rnn_dim), dtype=np.float32)

    attention_weights = np.zeros((B, MAX_TIME), dtype=np.float32)
    attention_weights_cum = np.zeros((B, MAX_TIME), dtype=np.float32)
    attention_context = np.zeros((B, self.encoder_embedding_dim), dtype=np.float32)

    return (
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
    )


def get_go_frame(self, memory):
    B = memory.shape[0]
    decoder_input = np.zeros((B, self.n_mel_channels * self.n_frames_per_step), dtype=np.float32)
    return decoder_input


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
    # (T_out, B) -> (B, T_out)
    alignments = np.stack(alignments).transpose((1, 0, 2, 3))
    # (T_out, B) -> (B, T_out)
    # Add a -1 to prevent squeezing the batch dimension in case
    # batch is 1
    gate_outputs = np.stack(gate_outputs).squeeze(-1).transpose((1, 0, 2))
    # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
    mel_outputs = np.stack(mel_outputs).transpose((1, 0, 2, 3))
    # decouple frames per step
    mel_outputs = mel_outputs.reshape(mel_outputs.shape[0], -1, self.n_mel_channels)
    # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
    mel_outputs = mel_outputs.transpose((0, 2, 1))

    return mel_outputs, gate_outputs, alignments


# only numpy operations
def test_inference(encoder, decoder_iter, postnet):
    parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.").to("cpu")
    sequences, sequence_lengths = parsed, np.array([parsed.size(1)])

    print("Running Tacotron2 Encoder")
    inputs = {"seq": sequences.numpy(), "seq_len": sequence_lengths}
    memory, processed_memory, _ = encoder.run(None, inputs)

    print("Running Tacotron2 Decoder")
    mel_lengths = np.zeros([memory.shape[0]], dtype=np.int32)
    not_finished = np.ones([memory.shape[0]], dtype=np.int32)
    mel_outputs, gate_outputs, alignments = [], [], []
    gate_threshold = 0.5
    max_decoder_steps = 1000
    first_iter = True

    (
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
    ) = initialize_decoder_states(spec_generator.decoder, memory)

    decoder_input = get_go_frame(spec_generator.decoder, memory)

    while True:
        inputs = {
            "decoder_input": decoder_input,
            "attention_hidden": attention_hidden,
            "attention_cell": attention_cell,
            "decoder_hidden": decoder_hidden,
            "decoder_cell": decoder_cell,
            "attention_weights": attention_weights,
            "attention_weights_cum": attention_weights_cum,
            "attention_context": attention_context,
            "memory": memory,
            "processed_memory": processed_memory,
        }
        (
            mel_output,
            gate_output,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
        ) = decoder_iter.run(None, inputs)

        if first_iter:
            mel_outputs = [np.expand_dims(mel_output, 2)]
            gate_outputs = [np.expand_dims(gate_output, 2)]
            alignments = [np.expand_dims(attention_weights, 2)]
            first_iter = False
        else:
            mel_outputs += [np.expand_dims(mel_output, 2)]
            gate_outputs += [np.expand_dims(gate_output, 2)]
            alignments += [np.expand_dims(attention_weights, 2)]

        dec = np.less(sigmoid(gate_output), gate_threshold)
        dec = np.squeeze(dec, axis=1)
        not_finished = not_finished * dec
        mel_lengths += not_finished

        if not_finished.sum() == 0:
            print("Stopping after ", len(mel_outputs), " decoder steps")
            break
        if len(mel_outputs) == max_decoder_steps:
            print("Warning! Reached max decoder steps")
            break

        decoder_input = mel_output

    mel_outputs, gate_outputs, alignments = parse_decoder_outputs(
        spec_generator.decoder, mel_outputs, gate_outputs, alignments
    )

    print("Running Tacotron2 PostNet")
    inputs = {"mel_spec": mel_outputs}
    mel_outputs_postnet = postnet.run(None, inputs)

    return mel_outputs_postnet


vocoder = HifiGanModel.from_pretrained(model_name="tts_en_hifigan").to("cpu")
vocoder.eval()
vocoder.export("vocoder.onnx")

spec_generator = Tacotron2Model.from_pretrained("tts_en_tacotron2").to("cpu")
spec_generator.eval()
spec_generator.export("en.onnx")

# Load encoder/decoder/postnet from onnx files
encoder = ort.InferenceSession("tacotron2encoder-en.onnx")
decoder = ort.InferenceSession("tacotron2decoder-en.onnx")
postnet = ort.InferenceSession("tacotron2postnet-en.onnx")

mel = test_inference(encoder, decoder, postnet)

# Use vocoder to get raw audio from spectrogram
hifi = ort.InferenceSession("vocoder.onnx")
audio = hifi.run(None, {"spec": mel[0]})
audio = audio[0][0, 0, :]
sf.write("speech.wav", audio, 22050, format="WAV")
