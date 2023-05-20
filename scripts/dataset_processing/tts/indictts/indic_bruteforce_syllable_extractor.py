import json
import string

data_dir = "./data/indic_tts"
lang_id = "hi"
target_file = f"{data_dir}/syllables_{lang_id}.json"
symbols = {"ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ", "்", "ஃ"}


def generate_grapheme_clusters(text, lang_symbols):
    clusters = []
    current_cluster = ""
    for i, char in enumerate(text):
        if char.isdigit() or char in string.punctuation or char in string.whitespace:
            continue
        if i == 0:
            current_cluster += char
        else:
            if char in lang_symbols:
                current_cluster += char
            else:
                clusters.append(current_cluster)
                current_cluster = char
    clusters.append(current_cluster)
    return clusters


with open(f"{data_dir}/train_manifest.json", 'r') as f:
    text = f.readlines()

text = [json.loads(x) for x in text]
all_text = " ".join(x["text"] for x in text)

syllables = generate_grapheme_clusters(all_text, symbols)

with open(target_file, 'w') as f:
    f.write(json.dumps(sorted(list(set(syllables)))))
