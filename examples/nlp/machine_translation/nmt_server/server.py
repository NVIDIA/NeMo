import argparse
import math
import os
import re
import time
from concurrent import futures

import api.nmt_pb2 as nmt
import api.nmt_pb2_grpc as nmtsrv
import grpc
import torch

import nemo.collections.nlp as nemo_nlp
from nemo.utils import logging

torch.set_grad_enabled(False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, action='append', help="")
    parser.add_argument("--port", default=50052, type=int, required=False)
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--beam_size", type=int, default=1, help="")
    parser.add_argument("--len_pen", type=float, default=0.6, help="")
    parser.add_argument("--max_delta_length", type=int, default=5, help="")

    args = parser.parse_args()
    return args


def batches(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class JarvisTranslateServicer(nmtsrv.JarvisTranslateServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self, model_paths, beam_size=1, len_pen=0.6, max_delta_length=5, batch_size=256):
        self._models = {}
        self._beam_size = beam_size
        self._len_pen = len_pen
        self._max_delta_length = max_delta_length
        self._batch_size = batch_size

        for model_path in model_paths:
            logging.info(f"Loading model {model_path}")
            self._load_model(model_path)
        logging.info("Models loaded. Ready for inference requests.")

    def _load_model(self, model_path):
        model_name, _ = os.path.splitext(os.path.basename(model_path))
        model_name = model_name.lower()
        if not re.match("^[\w]{2}-[\w]{2}$", model_name):
            logging.error("Model not named in language pair format src-target")
        if model_path.endswith(".nemo"):
            logging.info("Attempting to initialize from .nemo file")
            self._models[model_name] = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(
                restore_path=model_path
            )
        else:
            raise NotImplemented(f"Only support .nemo files, but got: {model_path}")

        self._models[model_name].beam_search.beam_size = self._beam_size
        self._models[model_name].beam_search.len_pen = self._len_pen
        self._models[model_name].beam_search.max_delta_length = self._max_delta_length

        if torch.cuda.is_available():
            self._models[model_name] = self._models[model_name].cuda()

    def TranslateText(self, request, context):
        logging.info(f"Request received w/ {len(request.texts)} utterances")
        results = []

        lang_pair = f"{request.source_language}-{request.target_language}".lower()
        if lang_pair not in self._models:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid lang pair {lang_pair} requeste")
            return nmt.TranslateTextResponse()

        request_strings = [x for x in request.texts]

        for batch in batches(request_strings, self._batch_size):
            batch_results = self._models[lang_pair].translate(text=batch, source_lang=None, target_lang=None)
            translations = [nmt.Translation(translation=x) for x in batch_results]
            results.extend(translations)

        return nmt.TranslateTextResponse(translations=results)


def serve():
    args = get_args()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = JarvisTranslateServicer(
        model_paths=args.model,
        beam_size=args.beam_size,
        len_pen=args.len_pen,
        batch_size=args.batch_size,
        max_delta_length=args.max_delta_length,
    )
    nmtsrv.add_JarvisTranslateServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:' + str(args.port))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
