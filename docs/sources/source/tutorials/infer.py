# this is NEMO's "core" package
import nemo
# this is NEMO's ASR collection of speech recognition related neural modules
import nemo_asr

# Path to the data on which you want to run inference
inference_manifest = "/home/okuchaiev/repos/gitlab-master/nemo/scripts/data" \
               "/dev_clean.json"

# Import Jasper model definition
import toml
jasper_model_definition = toml.load("/home/okuchaiev/repos/gitlab-master/nemo/examples/asr/tomls/jasper15x5SEP.toml")
labels = labels = jasper_model_definition['labels']['labels']

# Instantiate necessary neural modules
data_layer = nemo_asr.AudioToTextDataLayer(
    featurizer_config=jasper_model_definition['input'],
    manifest_filepath=inference_manifest,
    labels=labels, batch_size=1)
data_preprocessor = nemo_asr.AudioPreprocessing(
    **jasper_model_definition['input'])
jasper_encoder = nemo_asr.JasperEncoder(**jasper_model_definition)
jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024,
                                              num_classes=len(labels))
greedy_decoder = nemo_asr.GreedyCTCDecoder()

# Define inference DAG
audio_signal, audio_signal_len, transcript, transcript_len = data_layer()
processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal,
                                                           length=audio_signal_len)
encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
log_probs = jasper_decoder(encoder_output=encoded)
predictions = greedy_decoder(log_probs=log_probs)

from nemo_asr.helpers import process_evaluation_batch, word_error_rate
infer_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[log_probs, predictions, transcript, transcript_len],
    user_iter_callback=lambda x, y: process_evaluation_batch(
        x, y, labels=labels),
    user_epochs_done_callback=None,  # Unused
    eval_step=1,  # Unused
    tensorboard_writer=None  # Unused
)

neural_factory = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch)

optimizer = neural_factory.get_trainer(params={})
tensor_dict = optimizer.infer(
    callback=infer_callback,
    checkpoint_dir="/mnt/D2/JASPERNEMOCHECKPOINTS/15x5SEP/",
)

hypotheses = tensor_dict["predictions"]
references = tensor_dict["transcripts"]
wer = word_error_rate(hypotheses=hypotheses, references=references)

print("Greedy WER {:.2f}".format(wer*100))