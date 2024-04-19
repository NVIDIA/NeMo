## Introduction

We can run `mixtral_eval.py` to call mixtral api to give scores for the generated responses of two models.
Here we use [llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) as an example.

### Set up
Before running the script, we need to set up `NGC API KEY` for calling the foundation models on NVIDIA NGC. Once you set up your account on NGC, you can login in and go to [here](https://build.nvidia.com/mistralai/mixtral-8x7b-instruct) and click `Get API Key`. Save the key. That's it!


### Dataset

We first download [llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild).

```
git clone https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild
```
And download the [`rule.json` file](https://huggingface.co/spaces/LanguageBind/Video-LLaVA/blob/main/llava/eval/table/rule.json).


Notice the answer file in `llava-bench-in-the-wild` is consisted of rows of json string:
```
{"question_id": 0, "prompt": "What is the name of this famous sight in the photo?", "answer_id": "TeyehNxHw5j8naXfEWaxWd", "model_id": "gpt-4-0314", "metadata": {}, "text": "The famous sight in the photo is Diamond Head."}
```

You may also have your own response file as
```
{"response_id": 0, "response": "The famous sight in the photo is Diamond Head."}
```

Both formats are ok.

### Evaluation

Install package:
```
pip install shortuuid
```

Now you can run the script simply by
```
API_TOKEN=nvapi-<the api you just saved> python3 mixtral_eval.py --model-name-list gpt bard --media-type image  \
--question-file llava-bench-in-the-wild/questions.jsonl \  # the question file
--responses-list llava-bench-in-the-wild/answers_gpt4.jsonl llava-bench-in-the-wild/bard_0718.jsonl  \   # two answer files / response files
--answers-dir ./  \  # to save the answers
--context-file llava-bench-in-the-wild/context.jsonl \  # context file
--output ./output.json  # the generated mixtral reviews for the two models
```

You'll see the result like:
```
all 84.8 72.4
llava_bench_complex 77.0 69.0
llava_bench_conv 91.8 77.1
llava_bench_detail 91.3 73.2
```

Notice when you start a new comparison, you should remove the `output.json` file.