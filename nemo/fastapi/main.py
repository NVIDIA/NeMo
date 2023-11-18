from fastapi import FastAPI
from nemo.deploy import NemoQuery

app = FastAPI()


@app.get("/v1/completions/{model},{prompt},{max_tokens},{temperature},{top_p},{n},{stream},{stop},{frequency_penalty}")
async def completions_v1(
        model,
        prompt,
        max_tokens: int=512,
        temperature: float=1.0,
        top_p: float=0.5,
        n: int=1,
        stream: bool=False,
        stop="None",
        frequency_penalty: float=1.0,
):

    print('model: ', model)
    print('prompt: ', prompt)
    print('max_tokens: ', max_tokens)
    print('temperature: ', temperature)
    print('top_p: ', top_p)
    print('n: ', n)
    print('stream: ', stream)
    print('stop: ', stop)
    print('frequency_penalty: ', frequency_penalty)


    nq = NemoQuery(url="http://localhost:8002", model_name=model)
    output = nq.query_llm(
        prompts=[prompt],
        max_output_token=max_tokens,
        top_k=n,
        top_p=top_p,
        temperature=temperature,
    )


    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "stop": stop,
        "frequency_penalty": frequency_penalty,
    }


@app.get("/v1/chat/{model},{prompt},{max_tokens},{temperature},{top_p},{n},{stream},{stop},{frequency_penalty}")
async def chat_v1(
        model,
        prompt,
        max_tokens: int=512,
        temperature: float=1.0,
        top_p: float=0.5,
        n: int=1,
        stream: bool=False,
        stop="None",
        frequency_penalty: float=1.0,
):
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "stop": stop,
        "frequency_penalty": frequency_penalty,
    }
