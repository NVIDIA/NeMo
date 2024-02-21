import json
import time

import requests
from requests.packages.urllib3.util import Retry

ENDPOINT_URL = 'http://gpuwa.nvidia.com/dataflow/sandbox-pablo-usecase-inf/posting'  # your dataflow endpoint here


def postToNVDataFlowJson(json_data):
    headers = {'Content-Type': 'application/json', 'Accept-Charset': 'UTF-8'}
    Retry(total=5, allowed_methods=frozenset(['GET', 'POST']))  # retry a few times, dont give up after just 1 failure

    try:
        res = requests.post(ENDPOINT_URL, data=json_data, headers=headers)
        print(res.text)
    except Exception as err:
        print(Exception, err)


def postToNVDataFlow(dictionary):
    epoch_time_ms = int(time.time() * 1000)
    dictionary["ts_created"] = epoch_time_ms
    json_data = json.dumps(dictionary)

    print("JSON data to be posted:", json_data)
    postToNVDataFlowJson(json.dumps(dictionary))
