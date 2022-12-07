########################################################################
Example: Esperanto ASR using Mozilla Common Voice Dataset
########################################################################

Training an ASR model for a new language can be challenging because many specific features may differ depending on the language characteristics and amount of training data. At the moment, NeMo already has a detailed `example <https://github.com/NVIDIA/NeMo/blob/main/docs/source/asr/examples/kinyarwanda_asr.rst>`_ for Kinyarwanda ASR training. You can find all the information for getting the ASR model there. The current example aims to describe the ASR model training for Esperanto language and show some helpful practices that can improve recognition accuracy.

The example covers the next steps:

* Data preparation.
* Tokenization.
* Analysis of training parameters. 
* Training from scratch.
* Finetuning from pretrained models on other languages (English, Spanish, Italian).
* Finetuning from pretrained English SSL (`Self-supervised learning <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/ssl/intro.html?highlight=self%20supervised>`_) model.
* Model evaluation. 

**************************
Data preparation
**************************
Mozilla Common Voice provides a dataset for Esperanto language with about 1400 hours of validated data (general details of data corpuses creation can be found `here <https://arxiv.org/abs/1912.0667>`_). However, the final training dataset consists only of 250 hours because of the next rules – “The train, test, and development sets are bucketed such that any given speaker may appear in only one. This ensures that contributors seen at train time are not seen at test time, which would skew results. Additionally, repetitions of text sentences are removed from the train, test, and development sets of the corpus”. 

Download data
#################################

To get data manifests for Esperanto you can use the modefied NeMo `script <https://github.com/andrusenkoau/NeMo/blob/esperanto_example/docs/source/asr/examples/esperanto_asr/scripts/get_commonvoice_data_v2.py>`_.

.. code-block:: bash

    python ${NEMO_ROOT}/docs/source/asr/examples/esperanto_asr/scripts/get_commonvoice_data_v2.py \
      --data_root ${YOUR_DATA_ROOT}/esperanto/raw_data \
      --manifest_dir ${YOUR_DATA_ROOT}/esperanto/manifests \
      --log \
      --files_to_process 'test.tsv' 'dev.tsv' 'train.tsv' \
      --version cv-corpus-11.0-2022-09-21 \
      --language eo 

You will get the next data structure:

.. code-block:: bash

    ./esperanto
    ├── manifests
    │   ├── commonvoice_dev_manifest.json
    │   ├── commonvoice_test_manifest.json
    │   └── commonvoice_train_manifest.json
    └── raw_data
        ├── CV_unpacked
        │   └── cv-corpus-11.0-2022-09-21
        ├── dev
        │   └── wav
        ├── eo.tar.gz
        ├── test
        │   └── wav
        └── train
            └── wav
    ./esperanto/raw_data/CV_unpacked
    └── cv-corpus-11.0-2022-09-21
        └── eo
            ├── clips
            ├── dev.tsv
            ├── invalidated.tsv
            ├── other.tsv
            ├── reported.tsv
            ├── test.tsv
            ├── totalDur.sh
            ├── total_clips_duration.csv
            ├── train.tsv
            └── validated.tsv


Dataset preprocessing
#################################

Next, we must clear the text data from punctuation and various trash characters. In addition to deleting a standard set of elements (as in Kinyarwanda), you can build the frequency of characters encountered in the train set and add the rarest (occurring less than ten times) to the list for deletion. This approach will remove various garbage and leave only significant characters.

.. code-block:: python

  dev_manifest = f"{YOUR_DATA_ROOT}/esperanto/manifests/commonvoice_dev_manifest.json"
  test_manifest = f"{YOUR_DATA_ROOT}/esperanto/manifests/commonvoice_test_manifest.json"
  train_manifest = f"{YOUR_DATA_ROOT}/esperanto/manifests/commonvoice_train_manifest.json"

  def compute_char_counts(manifest):
      char_counts = {}
      with open(manifest, 'r') as fn_in:
          for line in tqdm(fn_in, desc="Compute counts.."):
              line = line.replace("\n", "")
              data = json.loads(line)
              text = data["text"]
              for word in text.split():
                  for char in word:
                      if char not in char_counts:
                          char_counts[char] = 1
                      else:
                          char_counts[char] += 1
      return char_counts

  char_counts = compute_char_counts(train_manifest)

  threshold = 10
  trash_char_list = []

  for char in char_counts:
      if char_counts[char] <= threshold:
          trash_char_list.append(char)

Let's check:

.. code-block:: python

  print(trash_char_list)

  ['é', 'ǔ', 'á', '¨', 'ﬁ', '=', 'y', '`', 'q', 'ü', '♫', '‑', 'x', '¸', 'ʼ', '‹', '›', 'ñ']

We will also check the data for anomalies. The simplest anomaly can be a problematic audio file. The text for it will look normal, but the audio file itself may be cut off or empty. One way to detect a problem is to check for char rate (number of chars per second). If the char rate is too high (more than 15 chars per second), then something is wrong with the file. It is better to filter such data from the training dataset in advance. Other problematic files can be filtered out after receiving the first trained model. We will consider this method at the end of our example.

.. code-block:: python

  import re

  def clear_data_set(manifest, char_rate_threshold=None):

      chars_to_ignore_regex = "[\.\,\?\:\-!;()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]"
      addition_ignore_regex = f"[{''.join(trash_char_list)}]"

      manifest_clean = manifest + '.clean'
      war_count = 0
      with open(manifest, 'r') as fn_in, \
          open(manifest_clean, 'w', encoding='utf-8') as fn_out:
          for line in tqdm(fn_in, desc="Cleaning manifest data"):
              line = line.replace("\n", "")
              data = json.loads(line)
              text = data["text"]
              if char_rate_threshold and len(text.replace(' ', '')) / float(data['duration']) > char_rate_threshold:
                  print(f"[WARNING]: {data['audio_filepath']} has char rate > 15 per sec: {len(text)} chars, {data['duration']} duration")
                  war_count += 1
                  continue
              text = re.sub(chars_to_ignore_regex, "", text)
              text = re.sub(addition_ignore_regex, "", text)
              data["text"] = text
              data = json.dumps(data, ensure_ascii=False)
              fn_out.write(f"{data}\n")
      print(f"[INFO]: {war_count} files were removed from manifest")

  clear_data_set(dev_manifest)
  clear_data_set(test_manifest)
  clear_data_set(train_manifest, char_rate_threshold=15)


Tarred dataset
#################################

The tarred dataset allows storing the dataset as large *.tar files instead of small separate audio files. It may speed up the training and minimizes the load on the network in the cluster.

The NeMo toolkit provides a `script <https://github.com/NVIDIA/NeMo/blob/main/scripts/speech_recognition/convert_to_tarred_audio_dataset.py>`_ to get tarred dataset.

.. code-block:: bash

    TRAIN_MANIFEST=${YOUR_DATA_ROOT}/esperanto/manifests/commonvoice_train_manifest.json.clean

    python ${NEMO_ROOT}/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
      --manifest_path=${TRAIN_MANIFEST} \
      --target_dir=${YOUR_DATA_ROOT}/esperanto/manifests/train_tarred_1bk \
      --num_shards=1024 \
      --max_duration=15.0 \
      --min_duration=1.0 \
      --shuffle \
      --shuffle_seed=1 \
      --sort_in_shards \
      --workers=-1

**************************
Tokenization
**************************

For Esperanto we use the standard `Byte-pair <https://en.wikipedia.org/wiki/Byte_pair_encoding>`_ encoding algorithm with 128, 512, and 1024 vocab size. It is worth noting that we have a relatively small training dataset (~250 hours). Usually, it is not enough data to train the best ARS model with a big vocab size (512 or 1024 BPE tokens). A smaller vocab size should be better in our case. We will check this statement further.

.. code-block:: bash

    VOCAB_SIZE=128
    
    python ${NEMO_ROOT}/scripts/tokenizers/process_asr_text_tokenizer.py \
      --manifest=${TRAIN_MANIFEST} \
      --vocab_size=${VOCAB_SIZE} \
      --data_root=${YOUR_DATA_ROOT}/esperanto/tokenizers \
      --tokenizer="spe" \
      --spe_type=bpe \  

**************************
Analysis of training parameters
**************************

Tuning of hyperparameters plays a huge role in training deep neural networks. The main list of parameters for training the standard ASR model in NeMo is presented in the `config file <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/conformer/conformer_ctc_bpe.yaml>`_ (general description of the `ASR configuration file <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/configs.html>`_). As an encoder, the `Conformer model <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#conformer-ctc>`_ is used here, the training parameters for which are already well configured based on the training English models. However, the set of optimal parameters may differ for a new language. In this section, we will look at the set of simple parameters that can improve recognition quality for a new language without digging into the Conformer model too much.

Batch size
#################################
As a local batch size we use 32 per GPU (V100). However, a large global batch size is usually required for stable model training since it allows to average gradients over a more significant number of training examples to smooth out outliers. The preferred global batch size is between 512 and 2048. To get such a number we suggest to use the `accumulate_grad_batches <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/conformer/conformer_ctc_bpe.yaml#L173>`_ parameter to artificially increase the size of the global batch and get the averaged gradient. As a result, the size of the global batch will be equal to *local_batch * num_gpu * accumulate_grad_batches*.

Scheduler
#################################
By default, the Conformer model in NeMo uses Noam as a learning rate scheduler. However, it has at least one disadvantage - the peak learning rate depends on the size of the model attention, the size of the global batch, and the number of warmup steps. The learning rate value itself for the optimizer is set in the config as some abstract number that will not be shown in reality. In order to still understand how the scheduler will look like, it is better to plot it in advance before training. You also can use a more understandable CosineAnealing scheduler.

Warmup steps
#################################
Number of warpup steps determines how quickly the scheduler will reach the peak learning rate during model training. One step equals a global batch size. If you increase the learning rate too fast, the model may diverge. The recommended number of steps is 8000-10000. If your model diverges, then you can try increasing this parameter.

Now we can plot our learning rate for CosineAnnealing schedule:

.. code-block:: python

    import nemo
    import torch
    import matplotlib.pyplot as plt

    # params:
    train_files_num = 144000     # number of training audio_files
    global_batch_size = 1024     # local_batch * gpu_num * accum_gradient
    num_epoch = 300
    warmup_steps = 10000
    config_learning_rate = 1e-3

    steps_num = int(train_files_num / global_batch_size * num_epoch)
    print(f"steps number is: {steps_num}")

    optimizer = torch.optim.SGD(model.parameters(), lr=config_learning_rate)
    scheduler = nemo.core.optim.lr_scheduler.CosineAnnealing(optimizer,
                                                             max_steps=steps_num,
                                                             warmup_steps=warmup_steps,
                                                             min_lr=1e-6)
    lrs = []

    for i in range(steps_num):
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        scheduler.step()

    plt.plot(lrs)

.. image:: ./images/CosineAnnealing_scheduler.png
    :align: center
    :alt: NeMo CosineAnnealing scheduler.
    :width: 500px
        
Precision
#################################
By default, it is recommended to use half-precision (FP16 for V100 and BF16 for A100 GPU) for ASR model training in NeMo. This allows you to speed up the training process almost twice. However, the transition to half-precision sometimes has problems with the convergence of the model. At an unexpected moment, the metrics can explode. In order to eliminate the influence of half-precision on such a problem, please check the training in FP32.

**************************
Training
**************************

We use three main scenarios for ASR model training:

* Training from scratch.
* Finetuning from already trained ASR models on other languages (English, Spanish, Italian).
* Finetuning from an English SSL (`Self-supervised learning <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/ssl/intro.html?highlight=self%20supervised>`_) model.

For the training of the `Conformer-CTC <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#conformer-ctc>`_ model, we use `speech_to_text_ctc_bpe.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py>`_ with the default config `conformer_ctc_bpe.yaml <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/conf/conformer/conformer_ctc_bpe.yaml>`_. Here you can see the example of how to run this training:

.. code-block:: bash

    TOKENIZER=${YOUR_DATA_ROOT}/esperanto/tokenizers/tokenizer_spe_bpe_v128
    TRAIN_MANIFEST=${YOUR_DATA_ROOT}/esperanto/manifests/train_tarred_1bk/tarred_audio_manifest.json
    TARRED_AUDIO_FILEPATHS=${YOUR_DATA_ROOT}/esperanto/manifests/train_tarred_1bk/audio__OP_0..1023_CL_.tar # "_OP_0..1023_CL_" is the range for the banch of files audio_0.tar, audio_1.tar, ..., audio_1023.tar
    DEV_MANIFEST=${YOUR_DATA_ROOT}/esperanto/manifests/commonvoice_dev_manifest.json.clean
    TEST_MANIFEST=${YOUR_DATA_ROOT}/esperanto/manifests/commonvoice_test_manifest.json.clean

    python ${NEMO_ROOT}/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
    --config-path=../conf/conformer/ \
    --config-name=conformer_ctc_bpe \
    exp_manager.name="Name of our experiment" \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.exp_dir=results/ \
    model.tokenizer.dir=$TOKENIZER \
    model.train_ds.is_tarred=true \
    model.train_ds.tarred_audio_filepaths=$TARRED_AUDIO_FILEPATHS \
    model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
    model.validation_ds.manifest_filepath=$DEV_MANIFEST \
    model.test_ds.manifest_filepath=$TEST_MANIFEST

Training parameters:

+----------------------------------+------------------------------------------------+
| Name                             |Value                                           |
+==================================+================================================+
|Tokenization                      |BPE 128/512/1024                                |
+----------------------------------+------------------------------------------------+
|Model                             |Conformer-CTC-large                             |
+----------------------------------+------------------------------------------------+
|Optimizer                         |AdamW, weight_decay 1e-3, LR 1e-3.              |
+----------------------------------+------------------------------------------------+
|Scheduler                         |CosineAnnealing, warmup_steps 10000, min_lr 1e-6|
+----------------------------------+------------------------------------------------+
|Batch                             |32 local, 1024 global (2 grad accumulation)     |
+----------------------------------+------------------------------------------------+
|Precision                         |FP16                                            |
+----------------------------------+------------------------------------------------+
|GPUs                              | 16 V100                                        |
+----------------------------------+------------------------------------------------+

The following table provides the results for training Esperanto Conformer-CTC-large model from scratch with different BPE vocab size.

+----------------------------------+----------+------------+-------------+
| Training mode                    | BPE size | DEV, WER % | TEST, WER % |
+==================================+==========+============+=============+
|                                  |    128   |     3.96   |     6.48    |
+                                  +----------+------------+-------------+
| From scratch                     |    512   |     4.62   |     7.31    |
+                                  +----------+------------+-------------+
|                                  |   1024   |     5.81   |     8.56    |
+----------------------------------+----------+------------+-------------+

The results show that BPE size 128 provides the lowest WER values. This may be because we have a small amount of training data (~250 hours), which is insufficient to train models with larger BPE vocab sizes. 

For finetuning from already trained ASR models, we use three different models:

* Esnglish `stt_en_conformer_ctc_large <https://huggingface.co/nvidia/stt_en_conformer_ctc_large>`_ (several thousand hours of English speech). 
* Spanish `stt_es_conformer_ctc_large <https://huggingface.co/nvidia/stt_es_conformer_ctc_large>`_ (1340 hours of Spanish speech).
* Italian `stt_it_conformer_ctc_large <https://huggingface.co/nvidia/stt_it_conformer_ctc_large>`_ (487 hours of Italian speech).

To finetune a model with the same vocab size, just set the desired model via the *init_from_pretrained_model* parameter:

.. code-block:: bash

    +init_from_pretrained_model=${PRETRAINED_MODEL_NAME}

as it done in the Kinyarwanda example. If the size of the vocab differs from the one presented in the pretrained model, you need to change the vocab manually as done in the `finetuning tutorial <https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb>`_:

.. code-block:: python

    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(f"nvidia/{PRETRAINED_MODEL_NAME}", map_location='cpu')
    model.change_vocabulary(new_tokenizer_dir=TOKENIZER, new_tokenizer_type="bpe")
    model.encoder.unfreeze()
    model.save_to(f"{save_path}")


There is no need to change anything for the SSL model, it will replace the vocab itself. However, you will need to first download this model and set it through another parameter *init_from_nemo_model*:

.. code-block:: bash

    ++init_from_nemo_model=${PRETRAINED_MODEL} \

As the SSL model, we use `ssl_en_conformer_large <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/ssl_en_conformer_large>`_ which is trained using LibriLight corpus (~56k hrs of unlabeled English speech).
All models for finetuning are available on `Nvidia Hugging Face <https://huggingface.co/nvidia>`_ or `NGC <https://catalog.ngc.nvidia.com/models>`_ repo. 

The following table shows all results for finetuning from pretrained models for the Conformer-CTC-large model and compares them with the model that was obtained by training from scratch (here we use BPE size 128 for all the models because it gives the best results).

+----------------------------------+------------+-------------+
| Training mode                    | DEV, WER % | TEST, WER % |
+==================================+============+=============+
| From scratch                     |     3.96   |     6.48    |
+----------------------------------+------------+-------------+
| Finetuning (English)             |     3.45   |     5.45    |
+----------------------------------+------------+-------------+
| Finetuning (Spanish)             |     3.40   |     5.52    |
+----------------------------------+------------+-------------+
| Finetuning (Italian)             |     3.29   |     5.36    |
+----------------------------------+------------+-------------+
| Finetuning (SSL English)         |  **2.90**  |   **4.76**  |
+----------------------------------+------------+-------------+

We can also look at the general trend of test WER decreasing in the training process using wandb plots (X - global step, Y - test WER):

.. image:: ./images/test_wer_wandb.png
    :align: center
    :alt: Test WER.
    :width: 800px

As you can see, the best way to get Esperanto ASR model is finetuning from the pretraind SSL model for English.


**************************
Decoding
**************************

At the end of the training, several checkpoints (usually 5) and one the best model (not always from the latest epoch) are stored in the model folder. Checkpoint averaging (script) can help to improve the final decoding accuracy. In our case, this did not improve the CTC models. However, for some RNNT models, it was possible to get an improvement in the range of 0.1-0.2% WER. To make averaging use the following command:

.. code-block:: bash

    python ${NEMO_ROOT}/scripts/checkpoint_averaging/checkpoint_averaging.py <your_trained_model.nemo>

For decoding you can use:

.. code-block:: bash

    python ${NEMO_ROOT}/examples/asr/speech_to_text_eval.py \
        model_path=${MODEL} \
        pretrained_name=null \
        dataset_manifest=${TEST_MANIFEST} \
        batch_size=${BATCH_SIZE} \
        output_filename=${OUTPUT_MANIFEST} \
        amp=False \
        use_cer=False)

You can use the Speech Data Explorer to analyze recognition errors, similar to the Kinyarwanda example.
After listening to files with an abnormally high WER (>50%), we found many problematic files with wrong transcriptions and cut or empty audio files in the dev and test sets.

.. code-block:: bash

    python ${NEMO_ROOT}/tools/speech_data_explorer/data_explorer.py <your_decoded_manifest_file>


**************************
Training data analysis
**************************

For an additional analysis of the training dataset, you can decode it using an already trained model. Train examples with a high error rate (WER > 50%) are likely to be problematic files. Removing them from the training set is preferred because a model can train text even for almost empty audio. We do not want this behavior from the ASR model.

