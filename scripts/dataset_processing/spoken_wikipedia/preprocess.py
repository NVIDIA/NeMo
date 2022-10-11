import argparse
import os
import re
from collections import defaultdict

import sox
from sox import Transformer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder", required=True, type=str, help="Input folder in which each subfolder contains an article"
)
parser.add_argument(
    "--destination_folder", required=True, type=str, help="Destination folder with audio and text subfolder"
)
args = parser.parse_args()


def replace_diacritics(text):
    text = re.sub(r"[éèëēêęěė]", "e", text)
    text = re.sub(r"[ãâāáäăâàąåạả]", "a", text)
    text = re.sub(r"[úūüùưûů]", "u", text)
    text = re.sub(r"[ôōóöõòő]", "o", text)
    text = re.sub(r"[ćçč]", "c", text)
    text = re.sub(r"[ïīíîıì]", "i", text)
    text = re.sub(r"[ñńňņ]", "n", text)
    text = re.sub(r"[țť]", "t", text)
    text = re.sub(r"[łľ]", "l", text)
    text = re.sub(r"[żžź]", "z", text)
    text = re.sub(r"[ğ]", "g", text)
    text = re.sub(r"[ř]", "r", text)
    text = re.sub(r"[ý]", "y", text)
    text = re.sub(r"[æ]", "ae", text)
    text = re.sub(r"[œ]", "oe", text)
    text = re.sub(r"[șşšś]", "s", text)
    return text


n = 0
duration = 0.0
for name in os.listdir(args.input_folder):
    n += 1

    audio_path = os.path.join(args.input_folder, name, "audio.ogg")
    if not os.path.exists(audio_path):
        ##  Some folders have multiple .ogg files, so we need to first combine them into one file. Example:
        ##  |── Universe
        ##  │   ├── aligned.swc
        ##  │   ├── audio1.ogg
        ##  │   ├── audio2.ogg
        ##  │   ├── audio3.ogg
        ##  │   ├── audio4.ogg
        ##  │   ├── audiometa.txt
        ##  │   ├── info.json
        ##  │   ├── wiki.html
        ##  │   ├── wiki.txt
        ##  │   └── wiki.xml

        multiple_ogg_files = []
        for i in range(1, 5):
            path = os.path.join(args.input_folder, name, "audio" + str(i) + ".ogg")
            if os.path.exists(path):
                multiple_ogg_files.append(path)
            else:
                break
        if len(multiple_ogg_files) == 0:
            continue
        elif len(multiple_ogg_files) == 1:
            os.system("cp " + multiple_ogg_files[0] + " " + audio_path)
        else:
            tmp_file_name = "ffmeg_inputs.txt"
            print("tmp_file_name=", tmp_file_name)
            with open(tmp_file_name, "w", encoding="utf-8") as tmp_file:
                for path in multiple_ogg_files:
                    tmp_file.write("file '" + path + "'\n")
            cmd = "ffmpeg -f concat -i " + tmp_file_name + " -c copy " + audio_path
            print("cmd=", cmd)
            os.system(cmd)

    # Then we need to convert .ogg to .wav
    output_wav_path = args.destination_folder + "/audio/" + str(n) + ".wav"
    tfm = Transformer()
    tfm.rate(samplerate=16000)
    tfm.channels(n_channels=1)
    if not os.path.exists(args.input_folder + "/" + name + "/wiki.txt"):
        print("wiki.txt does not exist in " + name)
        continue
    try:
        tfm.build(input_filepath=audio_path, output_filepath=output_wav_path)
        duration += sox.file_info.duration(output_wav_path)
    except Exception as e:
        print("Error in sox: " + name)
        print(e)
        continue

    # Then we need to clean the text
    out_text = open(args.destination_folder + "/text/" + str(n) + ".txt", "w", encoding="utf-8")
    with open(args.input_folder + "/" + name + "/wiki.txt", "r", encoding="utf-8") as f:
        for line in f:
            do_break = False
            line2 = line.strip()
            ref_parts = line2.split("<ref")
            for idx, s in enumerate(ref_parts):
                if idx != 0:
                    s = "<ref" + s
                if s.startswith("[[Image") and s.endswith("]]"):
                    continue
                if s.startswith("[[File") and s.endswith("]]"):
                    continue
                if s.startswith(":"):
                    continue
                if s.startswith("{|") or s.startswith("|}") or s.startswith("|") or s.startswith("!"):
                    continue
                if s.startswith("{{") and (s.endswith("}}") or "}}" not in s):
                    continue
                if s.startswith("{{pp-move"):
                    continue
                s = re.sub(r"\[\[Image\:[^\]]+\]\]", r"", s)
                s = re.sub(r"\[\[File\:[^\]]+\]\]", r"", s)
                s = re.sub(r"\[http[^\]]+\]", r"", s)
                s = re.sub(r"<math>[^<>]+</math>", r"", s)
                s = re.sub(r"<!\-\-.+\-\->", r"", s)  # <!--DashBot-->    can be inside <ref>
                s = re.sub(r"<ref>.+</ref>", r"", s)
                s = re.sub(r"<ref .+</ref>", r"", s)
                s = re.sub(r"<ref[^<>]+/>", r"", s)
                s = re.sub(r"<[^ <>]+>", r"", s)  # <sub>, <sup>, </u>
                if (
                    re.match(r"== *Notes *==", s)
                    or re.match(r"== *References *==", s)
                    or re.match(r"== *External links *==", s)
                    or re.match(r"== *See also *==", s)
                ):
                    do_break = True
                    break
                s = re.sub(r"{{convert\|(\d+)\|(\w+)\|[^}]+}}", r"\g<1> \g<2>", s)  # {{convert|7600|lb|kg}}
                s = re.sub(r"{{cquote\|", r"", s)
                s = re.sub(r"{{[^{}]+}}", r"", s)
                s = s.replace("{{", "").replace("}}", "")
                s = re.sub(r"(lang[^()]+)", r"", s)  # (lang-bn...)
                s = re.sub(r"==+", r"", s)
                s = re.sub(r"''+", r" ", s)  # remove multiple quotes
                s = re.sub(r" '", r" ", s)  # remove quote at the beginning
                s = re.sub(r"' ", r" ", s)  # remove quote at the end
                s = re.sub(r"[…\*]", r" ", s)
                s = re.sub(r"\\u....", r" ", s)  # remove unicode
                s = re.sub(r"&[^ ;&]+;", r"", s)  # &nbsp; &mdash;

                s = replace_diacritics(s)

                s = re.sub(r"\[\[[^\]]+\|([^\]]+)\]\]", r"\g<1>", s)  # if several variants, take the last one
                s = re.sub(r"\[\[([^\]]+)\]\]", r"\g<1>", s)

                out_text.write(s + "\n")
            if do_break:
                break
    out_text.close()
