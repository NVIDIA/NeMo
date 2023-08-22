import json
import requests
from pprint import pprint 
import time

batch_size = 1
port_num = 5554
headers = {"Content-Type": "application/json"}


def request_data(data):
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
			data=json.dumps(data),
			headers=headers)
    sentences = resp.json()['sentences']
    resp_dict = resp.json()
    return sentences, resp_dict


# response = conn.generate(
#   prompt="Q: what is the smallest natural number?",
#   model="gpt-43b-002",
#   stop=[],
#   tokens_to_generate=32,
#   temperature=1.0,
#   top_k=1,
#   top_p=0.0,
#   random_seed=0,
#   beam_search_diversity_rate=0.0,
#   beam_width=1,
#   repetition_penalty=1.0,
#   length_penalty=1.0,
# )

prompt_var="Q: what is the first letter of apple? "


def send_chat(prompt_var):
    if type(prompt_var) == list:
        prompt = prompt_var
    else:
        prompt = [prompt_var]
    data = {
        "sentences": prompt,
        "tokens_to_generate": 128,
        "temperature": 0.75,
        "add_BOS": True,
        "top_k": 0.99,
        "top_p": 0.0,
        "greedy": True,
        "all_probs": False,
        "repetition_penalty": 2.5,
        "min_tokens_to_generate": 2,
    }
    sentences, resp_dict = request_data(data)
    for sent in sentences:
        print(sent)
    return sentences

output = ""
while True:
    user_input = input("===> ")
    # user_input = f"{output}{user_input}"
    user_input_raw = [f"{user_input}"]
    user_input_list = []
    if type(user_input_raw) == list:
        for prompt in user_input_raw:
            prompt_qa = f"User: {prompt} \n\nAssistant: "
            # prompt_qa = f"Question: {prompt} \n\nAnswer: "
            user_input_list.append(prompt_qa)
    else:
        user_input_list = user_input_raw
    stt = time.time()
    output = send_chat(user_input_list)
    print(f"Time taken: {(time.time() - stt):.4f} sec for {len(user_input_list)} Questions")
