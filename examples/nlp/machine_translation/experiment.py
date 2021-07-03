from nemo.collections.nlp.models import MTEncDecModel

# To get the list of pre-trained models
MTEncDecModel.list_available_models()

# Download and load the a pre-trained to translate from English to Spanish
model = MTEncDecModel.from_pretrained("nmt_en_es_transformer12x2")

# Translate a sentence or list of sentences
translations = model.translate(["Hello!"], source_lang="en", target_lang="es")

print(translations)