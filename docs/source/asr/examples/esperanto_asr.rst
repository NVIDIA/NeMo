########################################################################
Example: Esperanto ASR using Mozilla Common Voice Dataset
########################################################################

Training an ASR model for a new language can be a challenging task, because there are many specific features which may differ depending on the language characteristics and amount of training data. At the moment NeMo already has a detailed example (link) for Kinyarwanda ASR training. You can find all the information for getting the ASR model there (read it first). The aim of the current example is to describe the ASR model training for Esperanto language and show some useful practices that can improve recognition accuracy. 

The example covers the next steps:

* Data preparation.
* Tokenization.
* Analysis of training parameters. 
* Training from scratch and finetuning.
* Model evaluation. 

**************************
Data preparation.
**************************
Mozilla Common Voice provides an dataset for Esperanto language with about 1400 hours of validated data (general details of data corpuses creation can be found `here <https://arxiv.org/abs/1912.0667>`_). However, the final training dataset consists only of 250 hours because the next rules – “The train, test, and development sets are bucketed such that any given speaker may appear in only one. This ensures that contributors seen at train time are not seen at test time, which would skew results. Additionally, repetitions of text sentences are removed from the train, test, and development sets of the corpus”. 

Download data:
#################################

To get data manifests for Esperanto you can use the modefied NeMo script `get_commonvoice_data.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/get_commonvoice_data.py>`_(change link to my file).

.. code-block:: bash

    python ${NEMO_ROOT}/scripts/dataset_processing/get_commonvoice_data_new.py \
      --data_root ${YOUR_DATA_ROOT}/esperanto/raw_data \
      --manifest_dir ${YOUR_DATA_ROOT}/esperanto/manifests \
      --log \
      --files_to_process 'test.tsv' 'dev.tsv' 'train.tsv' \
      --version cv-corpus-11.0-2022-09-21 \
      --language eo 

You will get next data structure:



Data preprocessing:
#################################

Next, we need to clear the text data from punctuation and various trash characters. In addition to deleting a common set of elements (as in Kinyarwanda), you can build the frequency of characters encountered in the train set and add the rarest (occurring less than 10 times) to the list for deletion. This approach will remove various garbage and leave really important characters.
We will also check the data for anomalies. The simplest anomaly can be a problematic audio file. The text for it will look normal, but the audio file itself may be cut off or empty. One way to detect a problem is to check for char rate (number of chars per second). If the char rate is too high (more than 15 chars per second), then something is clearly wrong with the file. It is better to filter such data from the training sample in advance. Other problematic files can be filtered out after receiving the first trained model. We will consider this method at the end of our tutorial.

.. code-block:: python

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

  char_counts = compute_char_counts(train_set)

  threshold = 10
  trash_char_list = []

  for char in char_counts:
      if char_counts[char] <= threshold:
          trash_char_list.append(char)

Let's check:

.. code-block:: python
  print(trash_char_list)

  ['é', 'ǔ', 'á', '¨', 'ﬁ', '=', 'y', '`', 'q', 'ü', '♫', '‑', 'x', '¸', 'ʼ', '‹', '›', 'ñ']
  
Now we need to clear our data:

.. code-block:: python

  import re

  def clear_data_set(manifest, char_rate_threshold=15, leav_cap_and_punct=False):

      chars_to_ignore_regex = "[\.\,\?\:\-!;()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]"
      addition_ignore_regex = f"[{''.join(trash_char_list)}]"

      manifest_clean = manifest + '.clean_all'
      war_count = 0
      with open(manifest, 'r') as fn_in, \
          open(manifest_clean, 'w', encoding='utf-8') as fn_out:
          for line in tqdm(fn_in, desc="Cleaning manifest data"):
              line = line.replace("\n", "")
              data = json.loads(line)
              text = data["text"]
              if len(text.replace(' ', '')) / float(data['duration']) > char_rate_threshold:
                  print(f"[WARNING]: {data['audio_filepath']} has char rate > 15 per sec: {len(text)} chars, {data['duration']} duration")
                  war_count += 1
                  continue
              text = re.sub(chars_to_ignore_regex, "", text)
              text = re.sub(addition_ignore_regex, "", text)
              data["text"] = text
              data = json.dumps(data, ensure_ascii=False)
              fn_out.write(f"{data}\n")
      print(f"[INFO]: {war_count} files were removed from manifest")

  clear_data_set(train_manifest)
  clear_data_set(dev_manifest)
  clear_data_set(test_manifest)


Tarred dataset:
#################################

Tarred dataset allows to store the dataset as large .tar files instead of small separate audio files. It may speed up the training and minimizes the load on the network in the cluster.

The NeMo toolkit provides a script to get tarred dataset.

.. code-block:: bash

    python ${NEMO_ROOT}/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
      --manifest_path=train_decoded_processed.json \
      --target_dir=train_tarred_1bk \
      --num_shards=1024 \
      --max_duration=15.0 \
      --min_duration=1.0 \
      --shuffle \
      --shuffle_seed=1 \
      --sort_in_shards \
      --workers=-1

**************************
Tokenization.
**************************

For Esperanto we use the standard Byte-pair encoding algorithm with 128, 512, and 1024 vocab size. It is worth noting that we have a relatively small training dataset (~250 hours). Usually it is not enough data to train the best ARS model with a big vocab size (512 or 1024 BPE tokens). Smaller vocab size should be better in our case. We will check this statement. 

.. code-block:: bash

    vocab_size=128
    python ${NEMO_ROOT}/scripts/tokenizers/process_asr_text_tokenizer.py \
      --manifest=dev_decoded_processed.json,train_decoded_processed.json \
      --vocab_size=$vocab_size \
      --data_root=tokenizer_bpe_maxlen_2 \
      --tokenizer="spe" \
      --spe_type=bpe \
      --spe_character_coverage=1.0 \
      --spe_max_sentencepiece_length=2 \
      --log

**************************
Analysis of training parameters. 
**************************

Tuning of hyper parameters plays a huge role in the training of deep neural networks. The main list of parameters for training the ASR model in NeMo is presented at the link. As an encoder, a Conformer model is used here, the training parameters for which are already well configured based on the training English models. However, for a new language, the set of optimal parameters may differ. In this section, we will look at the set of simple parameters that can improve the quality of recognition for a new language without digging into the Conformer model too much.


Scheduler:
#################################
By default, the Conformer model in NeMo uses Noam as a learning rate scheduler. However, it has at least one disadvantage - the peak learning rate depends on the size of the model attention, the size of the global batch, and the number of warmup steps. The learning rate value itself for the optimizer is set in the config as some abstract number that will not be shown in reality. In order to still understand how the schedule of the scheduler will look like, it is better to plot it in advance before training. Or use the more understandable CosineAnealing scheduler. The code for plotting the scheduler is shown below:
...

Warmup steps:
#################################
This parameter is responsible for how quickly your scheduler will reach the peak learning rate. One step is the size of your global batch (local batch * gpu_num * accum_gradient). If you increase the learning rate too quickly, the model may diverge. The recommended number of steps is 8000-10000. If your model diverges, then you can try increasing this parameter.

Batch size:
#################################
It is usually required to use a large global batch size, since it allows to average gradients over a larger number of training examples and to smooth out outliers. The good batch size is between 512 and 2048. Standard GPUs have 12-32 gigabytes of memory, which does not allow you to place such huge batches on them. Therefore, it is suggested to use grad_accamulation to artificially increase the size of the global batch and get the averaged gradient. As a local batch, it is not recommended to use a value greater than 32 (even if it fits on your GPU) because it can noticeably slow down the training speed. Most likely this is caused by the overhead of transferring data from RAM to the GPU memory ???.

Precision:
#################################
By default, for model training in NeMo, it is recommended to use half precision (FP16 for V100 and BF16 for A100 GPU). This allows you to speed up the training process almost twice. However, the transition to half-precision sometimes has problems with the convergence of the model. At an unexpected moment, the metrics can explode due to an overflow of one of the BN statistics. In order to eliminate the influence of half precision on such a problem, we advise you to check the training in FP32.

**************************
Training.
**************************

....

**************************
Decoding.
**************************

At the end of training, several checkpoints (usually 5) and one the best model (not always from the latest epoch) are stored in the model folder. Checkpoint averaging (script) can help to improve the final decoding accuracy. In our case, this did not give an improvement on the CTC models, however, for some RNNT models, it was possible to get an improvement in the range of 0.1-0.2% WER.

To analyze recognition errors, you can use the Speech Data Explorer, similar to the Kinyarwanda example. After listening to files with an abnormally high WER (>50%), problematic files with wrong transcriptions and cutted or empty audio files were found in the dev and test sets.

**************************
Bonus.
**************************

For additional analysis of the training dataset, you can decode it using an already trained model. Examples with high error rate (WER > 50%) are very likely to be problematic files. It is better to remove them from the training set. Sometimes a model is able to train text even for almost empty audio. Here you can see an example for this anomaly: …

