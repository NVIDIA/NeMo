from fastapi import FastAPI

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