import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

model_path = '/home/models/nemotron_hf'
tokenizer  = AutoTokenizer.from_pretrained(model_path)

#config = AutoConfig.from_pretrained("/home/models/nemotron_hf/config.json")

device = 'cuda'
dtype  = torch.bfloat16
model  = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=dtype, device_map=device)

#print(model)
#print(model.state_dict().keys())
# Prepare the input text
prompt = 'Complete the paragraph: our solar system is'
print(tokenizer.tokenize(prompt))
inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
#print(dir(tokenizer))
# Generate the output
outputs = model.generate(inputs, max_length=20)

# Decode and print the output
output_text = tokenizer.decode(outputs[0])
print(output_text)
