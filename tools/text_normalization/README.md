Text normalization system for english, e.g. 123 kg -> one hundred twenty three kilograms
Offers prediction and evaluation on text normalization data, e.g. google text normalization dataset
https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish.

Reaches 81% in sentence accuracy on output-00001-of-00100 of google text normalization dataset, 97.4% in token accuracy

Example evaluation run:
python run_evaluation.py  --input=./en_with_types/output-00001-of-00100 --normalizer nemo

Example prediction run:
python run_predicition.py  --input=text_data.txt --output=. --normalizer nemo