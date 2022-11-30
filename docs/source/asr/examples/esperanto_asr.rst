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
Mozilla Common Voice provides a dataset for Esperanto language with about 1400 hours of validated data (general details of data corpuses creation can be found `here <https://arxiv.org/abs/1912.0667>`_). However, the final training dataset consists only of 250 hours because the next rules – “The train, test, and development sets are bucketed such that any given speaker may appear in only one. This ensures that contributors seen at train time are not seen at test time, which would skew results. Additionally, repetitions of text sentences are removed from the train, test, and development sets of the corpus”. 

Download data:
#################################

To get data manifests for Esperanto you can use the modefied NeMo script `get_commonvoice_data.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/get_commonvoice_data.py>`_(change link to my file).

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


.. code-block:: python
  print(trash_char_list)

  ['é', 'ǔ', 'á', '¨', 'ﬁ', '=', 'y', '`', 'q', 'ü', '♫', '‑', 'x', '¸', 'ʼ', '‹', '›', 'ñ']

