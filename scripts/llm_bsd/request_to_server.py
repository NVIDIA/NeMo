import json
import requests
from pprint import pprint 
import time

batch_size = 1
port_num = 5550
headers = {"Content-Type": "application/json"}


def request_data(data):
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
			data=json.dumps(data),
			headers=headers)
    if type(resp.json()['sentences']) == str:
        raise ValueError("Error: {}".format(resp.json()['sentences']))
    
    sentences = resp.json()['sentences']
    resp_dict = resp.json()
    return sentences, resp_dict

def send_chat(prompt_var, tokens_to_generate=0):
    if type(prompt_var) == list:
        prompt = prompt_var
    else:
        prompt = [prompt_var]
        
    #         inference.compute_logprob=True \
    # inference.temperature=0.75 \
    # inference.greedy=True \
    # inference.top_k=0 \
    # inference.top_p=0.9 \
    # inference.all_probs=True \
    print(f"Prompt: {prompt}")
    data = {
        "sentences": prompt,
        "tokens_to_generate": tokens_to_generate,
        "temperature": 0.75,
        "add_BOS": True,
        "top_k": 0.99,
        "top_p": 0.0,
        "greedy": True,
        "compute_logprob": True,
        "all_probs": True,
        "repetition_penalty": 2.5,
        "min_tokens_to_generate": 2,
        "min_tokens_to_generate":1,
    }
    sentences, resp_dict = request_data(data)
    # for sent in sentences:
    #     print(sent)
    return sentences

output = ""
# prompt1="\[speaker1\]: and i i already got another apartment for when i moved out \[speaker0\]: oh you did \[speaker1\]: i had to put down like a deposit and um you know and pay the rent"
# prompt2="\[speaker1\]: and i i already got another apartment for when i moved out \[speaker0\]: oh you did \[speaker1\]: i had to put down like a deposit and um you know and pay the \[speaker0\]: rent"
prompt1="[speaker1]: and i i already got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and um you know and pay the rent"
prompt2="[speaker1]: and i i already got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and um you know and pay the [speaker0]: rent"
while True:
    user_input = input("===> ")
    # user_input = f"{output}{user_input}"
    user_input_raw = [prompt1, prompt2]
    # user_input_raw = [f"{user_input}"]
    user_input_list = []
    if type(user_input_raw) == list:
        for prompt in user_input_raw:
            prompt_qa = prompt
            # prompt_qa = f"User: {prompt} \n\nAssistant: "
            # prompt_qa = f"Question: {prompt} \n\nAnswer: "
            user_input_list.append(prompt_qa)
    else:
        user_input_list = user_input_raw
    stt = time.time()
    output = send_chat(user_input_list, tokens_to_generate=1)
    print(f"Time taken: {(time.time() - stt):.4f} sec for {len(user_input_list)} Questions")
