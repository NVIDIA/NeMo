Text normalization system for english, e.g. `123 kg` -> `one hundred twenty three kilograms`
Offers prediction and evaluation on text normalization data, e.g. [Google text normalization dataset](https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish).


Reaches 81% in sentence accuracy on output-00001-of-00100 of [Google text normalization dataset](https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish), 97.4% in token accuracy.

More details could be found in [this tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tools/Text_Normalization_Tutorial.ipynb).

Example evaluation run:
python run_evaluate.py  --input=./en_with_types/output-00001-of-00100 [--normalizer nemo]

Example prediction run:
python run_predict.py  --input=`INPUT_FILE` --output=`OUTPUT_FILE` [--normalizer nemo]