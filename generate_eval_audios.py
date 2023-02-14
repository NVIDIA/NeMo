import soundfile as sf
from nemo.collections.tts.models import InpainterModel
from nemo.collections.tts.models import HifiGanModel
import torch
import librosa
import sys
import os
from tqdm import tqdm

os.environ['DATA_CAP'] = '1'

args = sys.argv[1:]

try:
    checkpoint_path, recordings_path, output_dest = args
except:
    exit('usage: <checkpoint_path> <recordings_path> <output_dest>')

os.makedirs(output_dest, exist_ok=True)

model = InpainterModel.load_from_checkpoint(checkpoint_path)

# files_to_replacements = {
#     'test_1.wav': 'Would you like to speak with [joe burgarino] from [management team]?',
#     'test_2.wav': 'Would you like to speak with [linda toney] from [loss prevention]?',
#     'test_3.wav': 'Would you like to speak with [mary nelson] from [loss prevention]?',
#     'test_4.wav': 'Would you like to speak with [mary nelson] from [management team]?',
#     'test_5.wav': 'Would you like to speak with [joe burgarino] from [human resources]?',
#     'test_6.wav': 'Would you like to speak with [james brucken] from [human resources]?',
#     'test_7.wav': 'Ok that was [two blankets] and [two pillows] right?',
#     'test_8.wav': 'Ok that was a [toothbrush] and [a tooth] paste?',
#     'test_9.wav': 'Ok that was [three pillows] right?',
#     'test_10.wav': 'Good evening! Thanks for your loyalty as a [gold] member. How can I help?',
# }
# files_to_replacements = {
#   # "test01_p248_280_mic1.flac": "What would be the timetable for [withdrawal?]",
#   # "test02_p343_119_mic1.flac": "The hearing was expected to last [six weeks.]",
#   # "test03_p283_261_mic1.flac": "My wife and family are the [support system].",
#   # "test04_p288_188_mic1.flac": "The spokeswoman said the flat in [London was very small.]",
#   # "test05_p256_177_mic1.flac": "The conference is a chance for members to [contribute to the debate.]",
#   # "test06_p400_78.wav": "Would you like to speak to someone about [room service?]",
#   # "test07_p400_578.wav": "Our fitness center is on the nineteenth floor, and is open twenty-four seven. It can be accessed [using your room key.]",
#   # "test08_p400_298.wav": "Hi, thanks for calling the Orlando World Center Marriott! [How can I help you today]?",
#   # "test09_p400_336.wav": "All of our gift shops are in the Emerald Bay Market Plaza, just before the bridge to the convention center. Would you like to speak [with an outlet]?",
#   # "test10_p400_225.wav": "Would you like to speak to Siro Urban Italian Kitchen, Falls Pool Bar and Grill, Latitude and Longitude, The Lobby Lounge, High Velocity, or [the Mikado Japanese Steak House?]"
# }

# files_to_replacements = {
#     'p400_592.wav':  "While there are many [rental options in the area], we don't work with a specific partner.",
#     'p400_593.wav':  "All our current positions are available on our website: [careers dot marriott dot com]",
#     'p400_594.wav':  "Overnight valet parking is [sixty two dollars] per night plus tax.",
#     'p400_595.wav':  "Valet parking is [sixty two] dollars per night plus tax.",
#     'p400_596.wav':  "We have [valet parking] available at the hotel for sixty two dollars per night plus tax.",
#     'p400_597.wav':  "We do not offer [hourly] parking.",
#     'p400_598.wav':  "We welcome dogs or cats up to [forty pounds], and offer beds, bowls, and a doggie bag of treats and toys for free. If you have any questions, don't hesitate to reach out to the [front desk].",
# }

# files_to_replacements = {
# "LJ022-0023.wav": "The overwhelming majority of people in this country know how to sift the [wheat from the chaff] in what they hear and what they read.",
# "LJ005-0201.wav": "as is shown by the [report] of the Commissioners to inquire into the state of the municipal corporations in eighteen thirty-five.",
# "LJ001-0110.wav": "Even the Caslon type [when enlarged] shows great shortcomings in this respect:",
# "LJ003-0345.wav": "[All the committee could do] in this respect was to throw the responsibility on others.",
# "LJ018-0098.wav": "and recognized as one of the frequenters of the bogus law-stationers. [His arrest led to that of others.]",
# "LJ047-0044.wav": "Oswald was, however, [willing to discuss] his contacts with Soviet authorities. He denied having any involvement with Soviet intelligence agencies",
# "LJ031-0038.wav": "The first physician to see the President at Parkland Hospital was [Dr. Charles J. Carrico], a resident in general surgery.",
# "LJ048-0194.wav": "during the morning of [November twenty-two] prior to the motorcade.",
# }


files_to_replacements = {
    "2412/153954/2412_153954_000013_000000.wav": "We passed many cases, and at last came to one in which there were several clocks and [two or three] old watches.",
    "2412/153954/2412_153954_000007_000006.wav": "These two were first called out; and in about [a quarter of an hour] I was made to follow them, which I did in some fear, and with much curiosity.",
    "2412/153954/2412_153954_000005_000000.wav": "The [men] were as handsome as the women beautiful.",
    "2412/153954/2412_153954_000009_000003.wav": "They seemed concerned and uneasy as soon as they [got hold of it.]",
    "2412/153954/2412_153954_000011_000001.wav": "[He spoke to me] solemnly and sternly for two or three minutes.",
    "2412/153954/2412_153954_000007_000001.wav": "I will [spare the reader] any description of the town, and would only bid him think of Domodossola or Faido.",
    "2412/153954/2412_153954_000003_000004.wav": "[Glass] was plentiful in the better houses.",
}


def get_all_replacements(text):
    # assumes text is in correct format
    clean_str = ''
    replacements = []
    for ch in text:
        if ch == '[':
            new_replacement = clean_str + ch
            replacements = [new_replacement] + replacements
        elif ch == ']':
            # add it to the most recent replacement
            replacements[0] += ch
        else:
            clean_str += ch
            for i in range(len(replacements)):
                replacements[i] += ch

    return replacements


expected_sample_rate = 22050

vocoder = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")


def apply_repalcements(file_path, replacement_phrase, dest):
    data, sample_rate = sf.read(file_path)
    if sample_rate != expected_sample_rate:
        data = librosa.core.resample(
            data, orig_sr=sample_rate, target_sr=expected_sample_rate)

    original_spectrogram = model.make_spectrogram(data)
    replacements = get_all_replacements(replacement_phrase)

    spectrogram = original_spectrogram
    for replacement in replacements:
        full_replacement, partial_replacement, mcd_full, mcd_partial = model.regenerate_audio(
            spectrogram,
            replacement,
        )
        spectrogram = partial_replacement

    with torch.inference_mode():
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram.T.unsqueeze(0))

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    sf.write(dest, audio.to('cpu').numpy()[0], 22050, format='wav')


for filename, replacement_phrase in tqdm(files_to_replacements.items()):
    file_path = os.path.join(recordings_path, filename)
    dest = os.path.join(output_dest, filename)
    apply_repalcements(file_path, replacement_phrase, dest)
