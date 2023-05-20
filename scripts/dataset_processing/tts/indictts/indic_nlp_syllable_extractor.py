import json
import os

from indicnlp import common, loader
from indicnlp.syllable import syllabifier

## Clone the Indic NLP resources repository and set the path to INDIC_RESOURCES_ROOT

indic_nlp_resources_path = os.getenv("INDIC_RESOURCES_ROOT")
if not indic_nlp_resources_path:
    raise FileNotFoundError(
        "Indic NLP resources not found. Are you sure you added it to the environment variables under 'indic_nlp_resources'?"
    )
common.set_resources_path(indic_nlp_resources_path)
loader.load()

data_dir = "./data/indic_tts"
lang_id = "hi"
target_file = f"{data_dir}/{lang_id}/syllables_{lang_id}.json"


def syllable_extractor(text, lang_id):
    words = text.split()
    syllables = []
    for i, word in enumerate(words):
        syllables += syllabifier.orthographic_syllabify_improved(word, lang_id)
        syllables += [" "] if i != len(words) - 1 else []

    return syllables


with open(f"{data_dir}/{lang_id}/train_manifest.json", 'r') as f:
    text = f.readlines()

text = [json.loads(x) for x in text]
all_text = " ".join(x["text"] for x in text)

syllables = syllable_extractor(all_text, lang_id)

with open(target_file, 'w') as f:
    f.write(json.dumps(sorted(list(set(syllables)))))
