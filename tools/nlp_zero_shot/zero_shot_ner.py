import csv
import logging
from typing import List, Tuple, Dict
import argparse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from model_utils import load_model
from nemo.collections.nlp.models.dialogue.dialogue_zero_shot_slot_filling_model import DialogueZeroShotSlotFillingModel

logging.basicConfig(level=logging.INFO)


class Query(BaseModel):
    text: str = None
    entity_types: List[str] = None
    entity_descriptions: List[str] = None


class ZeroShotResponse(BaseModel):
    utterance_tokens: List[str] = None
    entities_dict: Dict[int, List[Tuple[int, int]]] = None
    slot_types: List[str] = None


app = FastAPI()


@app.post("/label/")
def predict(query: Query):
    types_descriptions = [_type + '\t' + _description
                          for _type, _description in zip(query.entity_types, query.entity_descriptions)]

    slot_class, iob_slot_class = model.predict(query.text, types_descriptions)
    slot_class, iob_slot_class = slot_class[0][1:-1], iob_slot_class[0][1:-1]
    utterance_tokens, slot_class, iob_slot_class = model.merge_subword_tokens_and_slots(query.text, slot_class, iob_slot_class)
    entities_dict = model.get_entities_start_and_end_dict(slot_class, utterance_tokens)
    print(entities_dict)

    types = query.entity_types

    if not types_descriptions:
        reader = csv.reader(model.slot_descriptions, delimiter="\t")
        types, descriptions = zip(*reader)
    return ZeroShotResponse(utterance_tokens=utterance_tokens, entities_dict=entities_dict, slot_types=types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a uvicorn web server for serving the zero shot model')
    parser.add_argument('--model_path', type=str, required=True, help='the zero-shot model path')
    parser.add_argument('--port', default=8082, type=int, help='please specify a server port')
    args = parser.parse_args()
    model: DialogueZeroShotSlotFillingModel = load_model(args.model_path)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level='debug')
