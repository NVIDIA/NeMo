Inverse Text Normalization system for english, e.g. `one hundred twenty three kilograms` -> `123 kg` 
Offers prediction and evaluation on text normalization data, e.g. [Google text normalization dataset](https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish).


Install dependencies:
bash setup.sh

Example prediction run:
python run_predict.py  --input=`INPUT_FILE` --output=`OUTPUT_FILE` [--verbose]

Example evaluation run:
python run_evaluate.py  --input=./en_with_types/output-00001-of-00100 [--cat CATEGORY] [--filter]
