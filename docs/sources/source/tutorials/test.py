# this is NEMO's "core" package
import nemo
# this is NEMO's ASR collection of speech recognition related neural modules
import nemo_asr

# We will use tensorboardX to keep track of train loss, eval wer etc.
from tensorboardX import SummaryWriter

tb_writer = SummaryWriter('jasper12x1SEP')

# Path to our training manifest
train_manifest = "/home/okuchaiev/repos/gitlab-master/nemo/scripts/data" \
                 "/train_clean_100.json"
# Path to our validation manifest
val_manifest = "/home/okuchaiev/repos/gitlab-master/nemo/scripts/data" \
               "/dev_clean.json"

# Jasper Model definition
import toml

# Here we will be using separable convolutions
# with 12 blocks (k=12 repeated once r=1 from the picture above)
jasper_model_definition = toml.load(
    "/home/okuchaiev/repos/gitlab-master/nemo/examples/asr/tomls"
    "/jasper12x1SEP.toml")
labels = jasper_model_definition['labels']['labels']

# Instantiate necessary neural modules
data_layer = nemo_asr.AudioToTextDataLayer(
    featurizer_config=jasper_model_definition['input'],
    manifest_filepath=train_manifest,
    labels=labels, batch_size=32)
data_preprocessor = nemo_asr.AudioPreprocessing(
    **jasper_model_definition['input'])
spec_augment = nemo_asr.SpectrogramAugmentation(**jasper_model_definition)
jasper_encoder = nemo_asr.JasperEncoder(**jasper_model_definition)
jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024,
                                              num_classes=len(labels))
ctc_loss = nemo_asr.CTCLossNM(num_classes=len(labels))
greedy_decoder = nemo_asr.GreedyCTCDecoder()

## Training DAG
audio_signal, audio_signal_len, transcript, transcript_len = data_layer()
processed_signal, processed_signal_len = data_preprocessor(
    input_signal=audio_signal,
    length=audio_signal_len)
aug_signal = spec_augment(input_spec=processed_signal)
encoded, encoded_len = jasper_encoder(audio_signal=aug_signal,
                                      length=processed_signal_len)

log_probs = jasper_decoder(encoder_output=encoded)
predictions = greedy_decoder(log_probs=log_probs)
loss = ctc_loss(log_probs=log_probs, targets=transcript,
                input_length=encoded_len, target_length=transcript_len)

# Validation DAG
# We need to instantiate additional data layer neural module
# for validation data
data_layer_val = nemo_asr.AudioToTextDataLayer(
    featurizer_config=jasper_model_definition['input'],
    manifest_filepath=val_manifest,
    labels=labels, batch_size=32)

audio_signal_v, audio_signal_len_v, transcript_v, transcript_len_v = \
    data_layer_val()
processed_signal_v, processed_signal_len_v = data_preprocessor(
    input_signal=audio_signal_v,
    length=audio_signal_len_v)
# Note that we are not using data-augmentation in validation DAG
encoded_v, encoded_len_v = jasper_encoder(audio_signal=processed_signal_v,
                                          length=processed_signal_len_v)
log_probs_v = jasper_decoder(encoder_output=encoded_v)
predictions_v = greedy_decoder(log_probs=log_probs_v)
loss_v = ctc_loss(log_probs=log_probs_v, targets=transcript_v,
                  input_length=encoded_len_v, target_length=transcript_len_v)

# These helper functions are needed to print and compute various metrics
# such as word error rate and log them into tensorboard
# they are domain-specific and are provided by NEMO's collections
from nemo_asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch

# Callback to track loss and print predictions during training
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensorboard_writer=tb_writer,
    # How to print loss to screen
    tensor_list2string=lambda x: str(x[0].item()),
    # How to print predictions and compute WER for train batches
    tensor_list2string_evl=lambda x: monitor_asr_train_progress(x,
                                                                labels=labels))

saver_callback = nemo.core.ModuleSaverCallback(
    save_modules_list=[jasper_encoder,
                       jasper_decoder],
    folder="./",
    # If set to x > 0 it will save modules every x steps
    # If set to = -1 it will only save once, after training is done
    step_frequency=-1)

# PRO TIP: while you can only have 1 train DAG, you can have as many
# val DAGs and callbacks as you want. This is useful if you want to monitor
# progress on more than one val dataset at once (say LibriSpeech dev clean
# and dev other)
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_v, predictions_v, transcript_v, transcript_len_v],
    # how to process evaluation batch - e.g. compute WER
    user_iter_callback=lambda x, y: process_evaluation_batch(
        x, y, labels=labels),
    # how to aggregate statistics (e.g. WER) for the evaluation epoch
    user_epochs_done_callback=lambda x: process_evaluation_epoch(x,
                                                                 tag="DEV-CLEAN"),
    eval_step=500,
    tensorboard_writer=tb_writer)

# Neural Module Factory manages training
# You will need to specify which backend to use
# Currently we only support PyTorch
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch)

# Optimizer
optimizer = neural_factory.get_trainer(
    params={"optimizer_kind": "novograd",
            "optimization_params": {"num_epochs": 50, "lr": 0.02,
                                    "weight_decay": 1e-4}})
# Run training
# Once this "action" is called data starts flowing along train and eval DAGs
# and computations start to happen
optimizer.train(tensors_to_optimize=[loss],
                callbacks=[train_callback, eval_callback, saver_callback],
                tensors_to_evaluate=[predictions, transcript, transcript_len])

