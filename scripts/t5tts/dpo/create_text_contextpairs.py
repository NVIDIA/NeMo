import json
import random
import copy
import argparse
import os

def corrupt_text(question_text):
    # randomly repeat word or delete a word from the question
    question_words = question_text.split(" ")
    if random.random() < 0.5:
        # repeat a word
        word_idx = random.randint(0, len(question_words) - 1)
        word = question_words[word_idx]
        # Repeat one occurence of the word
        question_text = question_text.replace(word, word + " " + word, 1)
    else:
        # delete a word
        word_idx = random.randint(0, len(question_words) - 1)
        word = question_words[word_idx]
        question_text = question_text.replace(word, "", 1)
    
    return question_text

def read_records(manifest_path):
    with open(manifest_path, 'r') as f:
        lines = f.readlines()
        records = []
        for line in lines:
            records.append(json.loads(line.strip()))
    return records

def write_records(fp, records):
    with open(fp, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print("Wrote {} records to: {}".format(len(records), fp))


def get_audio_filepath_from_codecpath(codec_path):
    # "target_codes_1638_84447_1638_84447_000105_000001.pt"
    # 1638/84447/1638_84447_000105_000001.wav
    codec_file_name = codec_path.split("/")[-1].split(".")[0]
    if "Lindy" in codec_file_name or "Rodney" in codec_file_name:
        # target_codes_Rodney_22khz_DROP_RODNEY_DROP_001185.pt
        # Rodney/44khz/DROP/RODNEY_DROP_000953.wav
        speaker = "RODNEY" if "Rodney" in codec_file_name else "LINDY"
        speaker_lowercase = "Rodney" if "Rodney" in codec_file_name else "Lindy"
        emotion_dir = codec_file_name.split("_22khz_")[1].split("_{}".format(speaker))[0] # DROP
        remaining_file_name = codec_file_name.split("_22khz_{}_".format(emotion_dir))[1]
        audio_file_path = "{}/22khz/{}/{}.wav".format(speaker_lowercase, emotion_dir, remaining_file_name)
        audio_file_path = os.path.join("/Data/RivaData/riva", audio_file_path)
        # import ipdb; ipdb.set_trace()
        # assert os.path.exists(audio_file_path), "File does not exist: {}".format(audio_file_path)
        return audio_file_path
    else:
        speaker_name = codec_file_name.split("target_codes_")[1].split("_")[0]
        chapter_name = codec_file_name.split("target_codes_")[1].split("_")[1]
        remaining_file_name = codec_file_name.split("target_codes_{}_{}_".format(speaker_name, chapter_name))[1]
        audio_file_path = "{}/{}/{}.wav".format(speaker_name, chapter_name, remaining_file_name)
        audio_file_path = os.path.join("/Data/LibriTTS/train-clean-360/", audio_file_path)
        # import ipdb; ipdb.set_trace()
        assert os.path.exists(audio_file_path), "File does not exist: {}".format(audio_file_path)
        return audio_file_path


parser = argparse.ArgumentParser()
parser.add_argument("--challenging_texts", type=str, default="/Data/challenging_texts_nemollm.txt")
parser.add_argument("--riva_manifest", type=str, default="/Data/CodecDatasets/speechllm_codecdatasets_new/manifests/rivaLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM.json")
parser.add_argument("--libri_manifest", type=str, default="/Data/CodecDatasets/speechllm_codecdatasets_new/manifests/libri360__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM.json")
parser.add_argument("--riva_textcontext_manifest", type=str, default="/Data/CodecDatasets/speechllm_codecdatasets_new/manifests/rivaLindyRodneyTextContext__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM.json")
parser.add_argument("--tts_records", type=bool, default=False)
parser.add_argument("--output_manifest", type=str, default="/Data/CodecDatasets/speechllm_codecdatasets_new/manifests/dpo_textcontext_pairs.json")
parser.add_argument("--nsamples_perpair", type=int, default=6)
args = parser.parse_args()

challenging_texts = args.challenging_texts
riva_manifest = args.riva_manifest
libri_manifest = args.libri_manifest
riva_textcontext_manifest = args.riva_textcontext_manifest
output_manifest = args.output_manifest

riva_records = read_records(riva_manifest)
libri_records = read_records(libri_manifest)
riva_textcontext_records = read_records(riva_textcontext_manifest)

# libri_records_longer_than_8 = [ record for record in libri_records if record['answer_duration'] > 8 ]
# riva_records_longer_than_8 = [ record for record in riva_records if record['answer_duration'] > 8 ]
libri_records_longer_than_2 = [ record for record in libri_records if record['answer_duration'] > 2 ]
riva_records_longer_than_2 = [ record for record in riva_records if record['answer_duration'] > 2 ]

with open(challenging_texts, 'r') as f:
    challenging_texts = f.readlines()

challenging_records = []
num_contexts_per_sample = 12
for challenging_text in challenging_texts:
    challenging_text = challenging_text.strip()
    for ci in range(num_contexts_per_sample):
        if ci >= num_contexts_per_sample - 2:
            # For last 20% of the challenging texts, make it more challenging by corrupting the text
            # Randomly drops a word or repeats a word
            print("Corrupting text: {}".format(challenging_text))
            challenging_text = corrupt_text(challenging_text)
            print("Corrupted text: {}".format(challenging_text))
        
        challenging_record_template = {
            'text': challenging_text,
            'target_audio_codes_path': None,
            'duration': 6.0,
            'context_audio_codes_path' : None,
            'context_audio_duration': None,
            'speaker': None
        }
        libri_challenging_record = copy.deepcopy(challenging_record_template)
        riva_challenging_record = copy.deepcopy(challenging_record_template)
        

        sample_libri_record = random.choice(libri_records)
        libri_challenging_record['context_audio_codes_path'] = sample_libri_record['context']
        libri_challenging_record['context_audio_filepath'] = get_audio_filepath_from_codecpath(sample_libri_record['context'])
        libri_challenging_record['context_audio_duration'] = sample_libri_record['context_duration']
        libri_challenging_record['speaker'] = sample_libri_record['speaker']
        libri_challenging_record['target_audio_codes_path'] = libri_challenging_record['context_audio_codes_path']
        libri_challenging_record['audio_filepath'] = libri_challenging_record['context_audio_filepath']

        sample_riva_record = random.choice(riva_records)
        riva_challenging_record['context_audio_codes_path'] = sample_riva_record['context']
        riva_challenging_record['context_audio_filepath'] = get_audio_filepath_from_codecpath(sample_riva_record['context'])
        riva_challenging_record['context_audio_duration'] = sample_riva_record['context_duration']
        riva_challenging_record['speaker'] = sample_riva_record['speaker']
        riva_challenging_record['target_audio_codes_path'] = riva_challenging_record['context_audio_codes_path']
        riva_challenging_record['audio_filepath'] = riva_challenging_record['context_audio_filepath']

        sample_riva_textcontext_record = random.choice(riva_textcontext_records)
        riva_textcontext_challenging_record = copy.deepcopy(challenging_record_template)
        del riva_textcontext_challenging_record['context_audio_codes_path']
        del riva_textcontext_challenging_record['context_audio_duration']
        riva_textcontext_challenging_record['text'] = challenging_text
        riva_textcontext_challenging_record['target_audio_codes_path'] = sample_riva_textcontext_record['answer']
        riva_textcontext_challenging_record['audio_filepath'] = get_audio_filepath_from_codecpath(sample_riva_textcontext_record['answer'])
        riva_textcontext_challenging_record['context_text'] = sample_riva_textcontext_record['context'].replace("TEXT CONTEXT: ", "Speaker and Emotion: ")

        challenging_records.append(libri_challenging_record)
        challenging_records.append(riva_challenging_record)
        if ci == 0:
            # dont need too many text context examples
            challenging_records.append(riva_textcontext_challenging_record)

# regular libri records 50% of the challenging records
libri_subset_records = random.sample(libri_records_longer_than_2, int(len(challenging_records)/2.0) )
libri_regular_records = []
for libri_subset_record in libri_subset_records:
    context_record = random.choice(libri_records)
    record = {
        'text': libri_subset_record['text'],
        'target_audio_codes_path': context_record['context'],
        'audio_filepath': get_audio_filepath_from_codecpath(context_record['context']),
        'duration': 6.0,
        'context_audio_codes_path' : context_record['context'],
        'context_audio_filepath': get_audio_filepath_from_codecpath(context_record['context']),
        'context_audio_duration': context_record['context_duration'],
    }
    libri_regular_records.append(record)
    

# regular riva records 20% of the challenging records
riva_subset_records = random.sample(riva_records_longer_than_2, int(len(challenging_records)/5.0))
riva_regular_records = []
for riva_subset_record in riva_subset_records:
    context_record = random.choice(riva_records)
    record = {
        'text': riva_subset_record['text'],
        'target_audio_codes_path': context_record['context'],
        'audio_filepath': get_audio_filepath_from_codecpath(context_record['context']),
        'duration': 6.0,
        'context_audio_codes_path' : context_record['context'],
        'context_audio_filepath': get_audio_filepath_from_codecpath(context_record['context']),
        'context_audio_duration': context_record['context_duration'],
    }
    riva_regular_records.append(record)

# riva textcontext records 5% of the challenging records
riva_textcontext_subset_records = random.sample(riva_textcontext_records, int(len(challenging_records)/20.0))
riva_textcontext_regular_records = []
for riva_textcontext_subset_record in riva_textcontext_subset_records:
    context_record = random.choice(riva_textcontext_records)
    record = {
        'text': riva_textcontext_subset_record['text'],
        'target_audio_codes_path': context_record['answer'],
        'audio_filepath': get_audio_filepath_from_codecpath(context_record['answer']),
        'duration': 6.0,
        'context_text' : context_record['context'].replace("TEXT CONTEXT: ", "Speaker and Emotion: "),
    }
    riva_textcontext_regular_records.append(record)

all_records = challenging_records + libri_regular_records + riva_regular_records + riva_textcontext_regular_records
random.shuffle(all_records)

# Repeate each record nsamples_perpair times
repeated_records = []
for record in all_records:
    for i in range(args.nsamples_perpair):
        repeated_records.append(record)

write_records(output_manifest, repeated_records)
write_records(output_manifest.replace(".json", "_240subset.json"), repeated_records[:240])