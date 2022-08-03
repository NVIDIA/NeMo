# slurp-evaluation

source: https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation


This package provides the code to use the SLU metrics proposed in *SLURP: A Spoken Language Understanding Resource Package* by Bastianelli, Vanzo, Swietojanski, and Rieser (EMNLP2020).

To install all the dependencies, simply run:
```shell script
$ pip install -r requirements.txt
```

The `Python` script `evaluate.py` allows to run the evaluation of a prediction file:

```shell script
$ python evaluate.py -h
usage: evaluate.py [-h] -g GOLD_DATA -p PREDICTION_FILE
                   [-d DISTANCE [DISTANCE ...]] [--load-gold] [--full]
                   [--errors] [--average AVERAGE]
                   [--table-layout TABLE_LAYOUT]

SLURP evaluation script

optional arguments:
  -h, --help            show this help message and exit
  -g GOLD_DATA, --gold-data GOLD_DATA
                        Gold data in SLURP jsonl format
  -p PREDICTION_FILE, --prediction-file PREDICTION_FILE
                        Predictions file
  --load-gold           When evaluating against gold transcriptions
                        (gold_*_predictions.jsonl), this flag must be true.
  --average AVERAGE     The averaging modality {micro, macro}.
  --full                Print the full results, including per-label metrics.
  --errors              Print TPs, FPs, and FNs in each row.
  --table-layout TABLE_LAYOUT
                        The results table layout {fancy_grid (DEFAULT), csv,
                        tsv}.
```

 * `GOLD_DATA` is the path to the jsonl file of the testing gold examples as they are provided in the release format of slurp. In the following, an example of line of the `test.jsonl` file:
```json
{"slurp_id": 4130,
"sentence": "is my reminder alarm set for dance class",
"sentence_annotation": "is my reminder alarm set for [event_name : dance class]",
"intent": "alarm_query",
"action": "query",
"tokens": [
  {"surface": "is", "id": 0, "lemma": "be", "pos": "VBZ"},
  {"surface": "my", "id": 1, "lemma": "-PRON-", "pos": "PRP$"},
  {"surface": "reminder", "id": 2, "lemma": "reminder", "pos": "NN"},
  {"surface": "alarm", "id": 3, "lemma": "alarm", "pos": "NN"},
  {"surface": "set", "id": 4, "lemma": "set", "pos": "VBN"},
  {"surface": "for", "id": 5, "lemma": "for", "pos": "IN"},
  {"surface": "dance", "id": 6, "lemma": "dance", "pos": "NN"},
  {"surface": "class", "id": 7, "lemma": "class", "pos": "NN"}
],
"scenario": "alarm",
"recordings": [
  {"file": "audio--1504192882-headset.wav", "wer": 0.0, "ent_wer": 0.0, "status": "correct"},
  {"file": "audio--1504192882.wav", "wer": 0.0, "ent_wer": 0.0, "status": "correct"},
  {"file": "audio--1504194663-headset.wav", "wer": 0.0, "ent_wer": 0.0, "status": "correct"},
  {"file": "audio--1504194663.wav", "wer": 0.0, "ent_wer": 0.0, "status": "correct"},
  {"file": "audio--1505405690-headset.wav", "wer": 0.0, "ent_wer": 0.0, "status": "correct"},
  {"file": "audio--1505405690.wav", "wer": 0.0, "ent_wer": 0.0, "status": "correct"},
  {"file": "audio-1497451768.wav", "wer": 0.0, "ent_wer": 0.0, "status": "correct"},
  {"file": "audio-1495371389-headset.wav", "wer": 0.25, "ent_wer": 1.0, "status": "correct"}
],
"entities": [
  {"span": [6, 7], "type": "event_name"}
]}
```
 * `PREDICTION_FILE` is the `jsonl` file containing predictions, where each line is provided in the following format:
```json
{"file": "audio--1504192882-headset.wav",
"scenario": "alarm",
"action": "query",
"entities": [
  {"type": "event_name", "filler": "dance class"}
]}
```
 * the flag `load-gold` must be set to `True` when evaluating against gold transcriptions and predicted NLU (`gold_*_predictions.jsonl`)
 * `AVERAGE` is the averaging modality (`micro`, `macro`)
 * the flag `full` allows to print the scores for each label
 * the flag `errors` prints TPs, FPs, and FNs for each row of the output label
 * `TABLE_LAYOUT` defined the output table format

To evaluate against predictions, run:
```shell script
python evaluate.py -g <PATH_TO_GOLD> -p <PATH_TO_PREDICTIONS>
```
