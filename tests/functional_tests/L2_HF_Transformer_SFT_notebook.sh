jupyter nbconvert --to script tutorials/llm/automodel/sft.ipynb --output _sft
sed -i "s#meta-llama/Llama-3.2-1B#/home/TestData/akoumparouli/hf_mixtral_2l/#g" tutorials/llm/automodel/_sft.py
sed -i "s/max_steps = 100/max_steps = 10/g" tutorials/llm/automodel/_sft.py
cp tutorials/llm/automodel/_sft.py /tmp/_sft.py
grep -iv push_to_hub /tmp/_sft.py > tutorials/llm/automodel/_sft.py
TRANSFORMERS_OFFLINE=1 python3 tutorials/llm/automodel/_sft.py
