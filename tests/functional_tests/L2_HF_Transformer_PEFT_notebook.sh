jupyter nbconvert --to script tutorials/llm/automodel/peft.ipynb --output _peft
sed -i "s#meta-llama/Llama-3.2-1B#/home/TestData/akoumparouli/hf_mixtral_2l/#g" tutorials/llm/automodel/_peft.py
sed -i "s/max_steps = 100/max_steps = 10/g" tutorials/llm/automodel/_peft.py
cp tutorials/llm/automodel/_peft.py /tmp/_peft.py
grep -iv push_to_hub /tmp/_peft.py > tutorials/llm/automodel/_peft.py
TRANSFORMERS_OFFLINE=1 python3 tutorials/llm/automodel/_peft.py
