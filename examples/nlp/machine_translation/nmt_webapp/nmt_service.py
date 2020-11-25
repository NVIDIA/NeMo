# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time

import torch
from flask import Flask, json, request

import nemo.collections.nlp as nemo_nlp
from nemo.utils import logging

model = None
api = Flask(__name__)

logging.info("Starting NMT service")
model = nemo_nlp.models.machine_translation.TransformerMTModel.restore_from(restore_path="TransformerMT.nemo")
if torch.cuda.is_available():
    logging.info("CUDA is available. Running on GPU")
    model = model.cuda()
else:
    logging.info("CUDA is not available. Defaulting to CPUs")
    logging.info("NMT service started")


@api.route('/translate', methods=['GET', 'POST'])
def get_translation():
    time_s = time.time()
    result = model.translate([request.args["text"]])
    duration = time.time() - time_s
    logging.info(
        f"Translated in {duration}. Input was: {request.args['text']} <############> Translation was: {result[0]}"
    )
    return json.dumps(result[0])


if __name__ == '__main__':
    api.run()
