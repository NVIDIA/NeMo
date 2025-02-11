# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from jinja2 import Template

from nemo.deploy.nlp import NemoQueryLLMPyTorch
from nemo.utils import logging


class TritonSettings(BaseSettings):
    _triton_service_port: int
    _triton_service_ip: str

    def __init__(self):
        super(TritonSettings, self).__init__()
        try:
            self._triton_service_port = int(os.environ.get('TRITON_PORT', 8000))
            self._triton_service_ip = os.environ.get('TRITON_HTTP_ADDRESS', '0.0.0.0')
        except Exception as error:
            logging.error("An exception occurred trying to retrieve set args in TritonSettings class. Error:", error)
            return

    @property
    def triton_service_port(self):
        return self._triton_service_port

    @property
    def triton_service_ip(self):
        return self._triton_service_ip


app = FastAPI()
triton_settings = TritonSettings()


class CompletionRequest(BaseModel):
    model: str
    prompt: str ='hello'
    messages: list[dict] = [{}]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.0
    top_k: int = 1
    logprobs: int = 1


@app.get("/v1/health")
def health_check():
    return {"status": "ok"}


@app.get("/v1/triton_health")
async def check_triton_health():
    """
    This method exposes endpoint "/triton_health" which can be used to verify if Triton server is accessible while running the REST or FastAPI application.
    Verify by running: curl http://service_http_address:service_port/v1/triton_health and the returned status should inform if the server is accessible.
    """
    triton_url = (
        f"http://{triton_settings.triton_service_ip}:{str(triton_settings.triton_service_port)}/v2/health/ready"
    )
    logging.info(f"Attempting to connect to Triton server at: {triton_url}")
    print("---triton_url---", triton_url)
    try:
        response = requests.get(triton_url, timeout=5)
        if response.status_code == 200:
            return {"status": "Triton server is reachable and ready"}
        else:
            raise HTTPException(status_code=503, detail="Triton server is not ready")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Triton server: {str(e)}")


@app.post("/v1/completions/")
async def completions_v1(request: CompletionRequest):
    try:
        url = f"http://{triton_settings.triton_service_ip}:{triton_settings.triton_service_port}"
        nq = NemoQueryLLMPyTorch(url=url, model_name=request.model)
        prompts = request.prompt
        if not isinstance(request.prompt, list):
            prompts = [request.prompt]
        output = nq.query_llm(
            prompts=prompts,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            compute_logprob=True if request.logprobs == 1 else False,
            max_length=request.max_tokens,
            init_timeout=300
        )

        # Convert NumPy arrays in output to lists
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj

        output_serializable = convert_numpy(output)
        ## #TODO Temp WAR
        output_serializable["choices"][0]["text"] = output_serializable["choices"][0]["text"][0][0]
        output_serializable["choices"][0]["logprobs"]["token_logprobs"] = output_serializable["choices"][0]["logprobs"]["token_logprobs"][0]
        output_serializable["choices"][0]["logprobs"]["top_logprobs"] = output_serializable["choices"][0]["logprobs"]["top_logprobs"][0]
        print("--output--", output_serializable)
        return output_serializable
    except Exception as error:
        logging.error(f"An exception occurred with the post request to /v1/completions/ endpoint: {error}")
        return {"error": "An exception occurred"}

# Define a function to apply the chat template
def apply_chat_template(messages, bos_token="<|startoftext|>", add_generation_prompt=False):
    from nemo.collections.llm.deploy.base import chat_template
    # Load the template
    template = Template(chat_template)

    # Render the template with the provided messages
    rendered_output = template.render(
        messages=messages,
        bos_token=bos_token,
        add_generation_prompt=add_generation_prompt
    )

    return rendered_output

@app.post("/v1/chat/completions/")
async def chat_completions_v1(request: CompletionRequest):
    try:
        url = f"http://{triton_settings.triton_service_ip}:{triton_settings.triton_service_port}"
        nq = NemoQueryLLMPyTorch(url=url, model_name=request.model)
        prompts = request.messages
        if not isinstance(request.messages, list):
            prompts = [request.messages]

        prompts_formatted = [apply_chat_template(prompts)]
        output = nq.query_llm(
            prompts=prompts_formatted,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            compute_logprob=True if request.logprobs == 1 else False,
            max_length=request.max_tokens,
            init_timeout=300
        )

        # Convert NumPy arrays in output to lists
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj

        output_serializable = convert_numpy(output)
        ## #TODO Temp WAR
        output_serializable["choices"][0]["text"] = output_serializable["choices"][0]["text"][0][0]
        output_serializable["choices"][0]["logprobs"]["token_logprobs"] = output_serializable["choices"][0]["logprobs"]["token_logprobs"][0]
        output_serializable["choices"][0]["logprobs"]["top_logprobs"] = output_serializable["choices"][0]["logprobs"]["top_logprobs"][0]
        print("--output--", output_serializable)
        return output_serializable
    except Exception as error:
        logging.error(f"An exception occurred with the post request to /v1/chat/completions/ endpoint: {error}")
        return {"error": "An exception occurred"}