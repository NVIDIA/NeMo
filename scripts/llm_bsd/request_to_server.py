import json
import requests
from pprint import pprint 
import time

batch_size = 1
port_num = 5501
# port_num = 5502
headers = {"Content-Type": "application/json"}


def request_data(data):
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
			data=json.dumps(data),
			headers=headers)
    if type(resp.json()) != dict:
        raise ValueError("Error: {}".format(resp.json()))
    
    resp_dict = resp.json()
    return resp_dict

def send_chat(prompt_var, tokens_to_generate=0):
    if type(prompt_var) == list:
        prompt = prompt_var
    else:
        prompt = [prompt_var]
        
    # inference.compute_logprob=True \
    # inference.temperature=0.75 \
    # inference.greedy=True \
    # inference.top_k=0 \
    # inference.top_p=0.9 \
    # inference.all_probs=True \
    # print(f"Prompt: {prompt}")
    data = {
        "sentences": prompt,
        "tokens_to_generate": tokens_to_generate,
        "temperature": 0.75,
        "add_BOS": True,
        "top_k": 0,
        "top_p": 0.9,
        "greedy": True,
        "compute_logprob": True,
        "all_probs": True,
        "repetition_penalty": 1.5,
        "min_tokens_to_generate":1,
    }
    # sentences, resp_dict = request_data(data)
    resp_dict = request_data(data)
    # for sent in sentences:
    #     print(sent)
    # return sentences, resp_dict
    return resp_dict

output = ""
# prompt1="[speaker1\]: and i i already got another apartment for when i moved out [speaker0\]: oh you did [speaker1\]: i had to put down like a deposit and um you know and pay the rent"
# prompt2="[speaker1\]: and i i already got another apartment for when i moved out [speaker0\]: oh you did [speaker1\]: i had to put down like a deposit and um you know and pay the [speaker0\]: rent"

prompt1="[speaker1]: and i i already got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and [speaker0] um you know and pay the rent"
prompt2="[speaker1]: and i i already got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and um you know and pay the [speaker0]: rent"
prompt3="[speaker1]: and i i already got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and um you know and pay the [speaker0]: rent"
prompt4="[speaker1]: and [speaker0] i i already [speaker1] got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and um you know and pay the [speaker0]: rent"

# prompt1_spk="User: [speaker0]: and i i already got another apartment for when i moved out [speaker1]: oh you did [speaker0]: i had to put down like a deposit and um you know and pay the \n [End of Dialogue] \n The next word is (rent). Which speaker spoke this word [speaker0] or [speaker1] ?\n\nAssistant:[speaker"
# prompt2_spk="User: [speaker1]: and i i already got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and um you know and pay the \n [End of Dialogue] \n The next word is (rent). Which speaker spoke this word [speaker0] or [speaker1] ?\n\nAssistant:[speaker"
# f"[speaker0]: i got another apartment for when i moved out [speaker1]: oh you did [speaker0]: i had to put down like a deposit and um you know and pay the [end of dialogue] \n Question: The next word is {NEXTWORD}. Who will speak {NEXTWORD} ? \n\nAnswer:[speaker",
# f"[speaker1]: i got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and um you know and pay the [end of dialogue] \n Question: The next word is {NEXTWORD}. Who will speak {NEXTWORD} ? \n\nAnswer:[speaker",
NEXTWORD = "(rent)"
SPEAKER0 = "[speaker0]:"
SPEAKER1 = "[speaker1]:"

user_input_raw = [
# f"User: {SPEAKER0} i already got another apartment for when i moved out {SPEAKER1}: oh you did {SPEAKER0} i had to put down like a deposit and um you know and pay the [end] \n Question: \n The next word is (rent). Which speaker will speak (rent) ? \n\nAssistant:[speaker",
# f"User: {SPEAKER1}: i already got another apartment for when i moved out {SPEAKER0} oh you did {SPEAKER1}: i had to put down like a deposit and um you know and pay the [end] \n Question: \n The next word is (rent). Which speaker will speak (rent) ? \n\nAssistant:[speaker",
# f"{SPEAKER0} i got another apartment for when i moved out {SPEAKER1} oh you did {SPEAKER0} i had to put down like a deposit and um you know and pay the [end] Question: The next word is {NEXTWORD}. Who spoke {NEXTWORD} ? \nAnswer:[speaker",
# f"{SPEAKER1} i got another apartment for when i moved out {SPEAKER0} oh you did {SPEAKER1} i had to put down like a deposit and um you know and pay the [end] Question: The next word is {NEXTWORD}. Who spoke {NEXTWORD} ? \nAnswer:[speaker",
f"User: {SPEAKER0} i already got another apartment for when i moved out {SPEAKER1}: oh you did {SPEAKER0} i had to put down like a deposit and um you know and pay the [end] \n Question: \n The next word is (rent). Which speaker will speak (rent) ? \n\nAssistant:[speaker",
f"User: {SPEAKER1}: i already got another apartment for when i moved out {SPEAKER0} oh you did {SPEAKER1}: i had to put down like a deposit and um you know and pay my [end] \n Question: \n The next word is (rent). Which speaker will speak (rent) ? \n\nAssistant:[speaker",
f"User: {SPEAKER1}: i already got another apartment for when i moved out {SPEAKER0} oh you did {SPEAKER1}: i had to put down like a deposit and um you know and pay landlord the [end] \n Question: \n The next word is (rent). Which speaker will speak (rent) ? \n\nAssistant:[speaker",
f"{SPEAKER0} i got another apartment for when i moved out {SPEAKER1} oh you did {SPEAKER0} i had to put down like a deposit and um you know i need to pay some [end] Question: The next word is {NEXTWORD}. Who spoke {NEXTWORD} ? \nAnswer:[speaker",
f"{SPEAKER1} i got another apartment for when i moved out {SPEAKER0} oh you did {SPEAKER1} i had to put down like a deposit and um hardly it is the [end] Question: The next word is {NEXTWORD}. Who spoke {NEXTWORD} ? \nAnswer:[speaker",
]

# prompt1_spk="User: [speaker0]: and i i already got another apartment for when i moved out [speaker1]: oh you did [speaker0]: you know and pay the \n Question: \n The next word is `rent`. Which speaker spoke this word [speaker0] or [speaker1] ? \n\nAssistant:[speaker"
# prompt2_spk="User: [speaker1]: and i i already got another apartment for when i moved out [speaker0]: oh you did [speaker1]: and um you know and pay the \n Question: \n The next word is `rent`. Which speaker spoke this word [speaker0] or [speaker1] ? \n\nAssistant:[speaker"

# prompt1_spk="[speaker0]: got another apartment for when i moved out [speaker1]: oh you did [speaker0]: i had to put down like a deposit and um you know and pay the \n [End of Dialogue] \n The next word is [rent]. Which speaker spoke this word [speaker0] or [speaker1] ?\n\n Answer:[speaker"
# prompt2_spk="[speaker1]: already got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and um you know and pay the \n [End of Dialogue] \n The next word is [rent]. Which speaker spoke this word [speaker0] or [speaker1] ?\n\n Answer:[speaker"
# prompt3_spk="[speaker1]: i already got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and um you know and pay the \n [End of Dialogue] \n The next word is [rent]. Which speaker spoke this word [speaker0] or [speaker1] ?\n\n Answer:[speaker"
# prompt4_spk="[speaker1]: and i already got another apartment for when i moved out [speaker0]: oh you did [speaker1]: i had to put down like a deposit and um you know and pay the \n [End of Dialogue] \n The next word is [rent]. Which speaker spoke this word [speaker0] or [speaker1] ?\n\n Answer:[speaker"

count = 0
total_time = 0
total_count_requests = 100
while count < total_count_requests:
    user_input = 2
    # user_input = f"{output}{user_input}"
    if user_input in [1, "1"]:
        tokens_to_generate = 1
        user_input_raw = [prompt1, prompt2, prompt3, prompt4]
    else:
        # tokens_to_generate = 4
        tokens_to_generate = 12
        # user_input_raw = [prompt1_spk, prompt2_spk, prompt3_spk, prompt4_spk]
        # user_input_raw = [prompt1_spk, prompt2_spk]
        user_input_raw = user_input_raw 
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
    resp_dict = send_chat(user_input_list, tokens_to_generate=tokens_to_generate)
    print(f"Time taken: {(time.time() - stt):.4f} sec for {len(user_input_list)} Questions, count = {count}/{total_count_requests}")
    count += 1
    eta = time.time() - stt
    total_time += eta
    # print(f"output = {resp_dict}")
    
print(f"Average time taken: {(total_time/count):.4f} sec for {len(user_input_list)} Questions")

while True:
    user_input = input("===> ")
    # user_input = f"{output}{user_input}"
    if user_input in [1, "1"]:
        tokens_to_generate = 1
        user_input_raw = [prompt1, prompt2, prompt3, prompt4]
    else:
        # tokens_to_generate = 4
        tokens_to_generate = 12
        # user_input_raw = [prompt1_spk, prompt2_spk, prompt3_spk, prompt4_spk]
        # user_input_raw = [prompt1_spk, prompt2_spk]
        user_input_raw = user_input_raw 
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
    resp_dict = send_chat(user_input_list, tokens_to_generate=tokens_to_generate)
    print(f"Time taken: {(time.time() - stt):.4f} sec for {len(user_input_list)} Questions")
    print(f"output = {resp_dict}")
    # import ipdb; ipdb.set_trace()