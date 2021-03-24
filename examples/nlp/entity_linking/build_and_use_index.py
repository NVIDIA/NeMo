from omegaconf import OmegaConf
from nemo.collections.nlp.models import EntityLinkingModel
from nemo.utils import logging


config_path="conf" 
config_name="medical_entity_linking_config_pubmed.yaml"
cfg = OmegaConf.load(f"{config_path}/{config_name}")
logging.info(f"\nConfig Params:\n{cfg.pretty()}")

logging.info("Restoring Model")
model = EntityLinkingModel(cfg.model)
model.load_index(cfg.index)
logging.info("Querying Index")

while True:
    query = input("enter index query: ")
    output = model.query_index(query)

    for concept in output:
        print(concept, output[concept])

