# This is where the an4/ directory will be placed.
# Change this if you don't want the data to be extracted in the current directory.
# --- Building Manifest Files --- #
import os
import pickle
import nemo
from ruamel.yaml import YAML
# NeMo's ASR collection
import nemo.collections.asr as nemo_asr
from nemo.utils.lr_policies import CosineAnnealing

data_dir = '/data/samsungSSD/NVIDIA/datasets/LibriSpeech/'
abs_dir=os.path.abspath(data_dir)
# NeMo's "core" package


labels = pickle.load(open(data_dir+'/count_spkr_labels.pkl','rb'))

train_manifest=abs_dir+'/sub_train_manifest.json'
dev_manifest=abs_dir+'/sub_dev_manifest.json'

# --- Loading Config --- #


# Parse config and pass to model building function
config_path = './configs/jasper_spkr2.yaml'
yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
    print("******\nLoaded config file.\n******")

# labels = params['labels']  # Vocab of tokens
sample_rate = params['sample_rate']

print(params)

# Create our NeuralModuleFactory, which will oversee the neural modules.
neural_factory = nemo.core.NeuralModuleFactory(
    log_dir=abs_dir +'/spkr_test1/',
    tensorboard_dir='cov_exp1',
    optimization_level="O1",
    create_tb_writer=True)

logging = nemo.logging 

# --- Instantiate Neural Modules --- #
batch_size=64
# Create training and test data layers (which load data) and data preprocessor
data_layer_train = nemo_asr.AudioToLabelDataLayer(
    manifest_filepath=train_manifest,
    sample_rate=sample_rate,
    labels=labels,
    batch_size=batch_size,
    **params['AudioToLabelDataLayer']['train'])  # Training datalayer

data_layer_test = nemo_asr.AudioToLabelDataLayer(
    manifest_filepath=dev_manifest,
    sample_rate=sample_rate,
    labels=labels,
    batch_size=batch_size,
    **params['AudioToLabelDataLayer']['eval'])   # Eval datalayer

data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
    sample_rate=sample_rate,
    **params['AudioToMelSpectrogramPreprocessor'])

# Create the Jasper_4x1 encoder as specified, and a CTC decoder
encoder = nemo_asr.JasperEncoder(**params['JasperEncoder'])

decoder = nemo_asr.JasperDecoderForSpkrClass(
    feat_in=params['JasperEncoder']['jasper'][-1]['filters'],
    num_classes=len(labels),
    covr=True)

xent_loss = nemo_asr.CrossEntropyLossNM()

# --- Assemble Training DAG --- #
audio_signal, audio_signal_len, label, label_len = data_layer_train()

processed_signal, processed_signal_len = data_preprocessor(
    input_signal=audio_signal,
    length=audio_signal_len)

encoded, encoded_len = encoder(
    audio_signal=processed_signal,
    length=processed_signal_len)

logits = decoder(encoder_output=encoded)
loss = xent_loss(
        logits=logits,
        labels=label)

# --- Assemble Validation DAG --- #
audio_signal_test, audio_len_test, label_test, label_len_test = data_layer_test()

processed_signal_test, processed_len_test = data_preprocessor(
    input_signal=audio_signal_test,
    length=audio_len_test)

encoded_test, encoded_len_test = encoder(
    audio_signal=processed_signal_test,
    length=processed_len_test)

logits_test = decoder(encoder_output=encoded_test)
loss_test = xent_loss(
    logits=logits_test,
    labels=label_test)

# --- Create Callbacks --- #

# We use these imports to pass to callbacks more complex functions to perform.
from nemo.collections.asr.helpers import monitor_classification_training_progress, \
    process_classification_evaluation_batch, process_classification_evaluation_epoch
from functools import partial

train_callback = nemo.core.SimpleLossLoggerCallback(
    # Notice that we pass in loss, predictions, and the transcript info.
    # Of course we would like to see our training loss, but we need the
    # other arguments to calculate the WER.
    tensors=[loss, logits, label],
    # The print_func defines what gets printed.
    print_func=partial(
        monitor_classification_training_progress,
        eval_metric=[1]),
    step_freq=500,
    get_tb_values=lambda x:[("train_loss", x[0])],
    tb_writer=neural_factory.tb_writer
    )

# We can create as many evaluation DAGs and callbacks as we want,
# which is useful in the case of having more than one evaluation dataset.
# In this case, we only have one.
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_test, logits_test, label_test],
    user_iter_callback=partial(
        process_classification_evaluation_batch, top_k=1),
    user_epochs_done_callback=process_classification_evaluation_epoch,
    eval_step=500,  # How often we evaluate the model on the test set
    tb_writer=neural_factory.tb_writer
    )

checkpoint_saver_callback = nemo.core.CheckpointCallback(
    folder=data_dir+'/spkr_checkpoints1',
    step_freq=1000  # How often checkpoints are saved
    )

if not os.path.exists(data_dir+'/spkr_checkpoints1'):
    os.makedirs(data_dir+'/spkr_checkpoints1')

# neural_factory.reset_trainer()
N= len(data_layer_train)
num_gpus=1
steps_per_epoch = int(N / (batch_size *num_gpus))
# --- Start Training! --- #
neural_factory.train(
    tensors_to_optimize=[loss],
    callbacks=[train_callback, eval_callback, checkpoint_saver_callback],
    optimizer='novograd',
    lr_policy=CosineAnnealing(20 * steps_per_epoch, warmup_steps=100),
    optimization_params={
        "num_epochs": 20, "lr": 0.01, "weight_decay": 1e-4
    })
