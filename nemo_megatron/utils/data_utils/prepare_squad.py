import os
import json
import time
from pathlib import Path
from .download_squad import download_squad

NEMO_MEGATRON_CI = os.getenv("NEMO_MEGATRON_CI", "False").lower() in ("true", "t", "1")

def prepare_squad_for_prompt_learning(data_dir, nemo_megatron_path):
    squad_dir = data_dir
    download_squad(squad_dir, ["v1.1"])
    squad_v1_dir = os.path.join(squad_dir, "v1.1")

    preprocess_script = nemo_megatron_path / "nemo_megatron/utils/data_utils/prompt_learning_squad_preprocessing.py"
    os.system(
        f"python3 {preprocess_script} "
        f"--data-dir={squad_v1_dir} "
    )


def prepare_squad_for_fine_tuning(data_dir):
    squad_dir = data_dir
    download_squad(squad_dir, ["v1.1", "xquad"])

    squad_v1_dir = os.path.join(squad_dir, "v1.1")
    squad_xquad_dir = os.path.join(squad_dir, "xquad")

    path2dev = {
        **{
            f"{squad_v1_dir}/train-v1.1.json": False,
            f"{squad_v1_dir}/dev-v1.1.json": True,
        },
        **{
            f"{squad_xquad_dir}/xquad.{lang}.json": True
            for lang in ["en", "es", "de", "el", "ru", "tr", "ar", "vi", "th", "zh", "hi"]
        },
    }

    for path, dev in path2dev.items():
        if not os.path.exists(f"{os.path.splitext(path)[0]}_src.txt") or \
                not os.path.exists(f"{os.path.splitext(path)[0]}_tgt.txt"):
            preprocess_squad_for_fine_tuning(
                fname=path,
                out_fname_prefix=os.path.splitext(path)[0],
                dev=dev,
            )


def preprocess_squad_for_fine_tuning(fname, out_fname_prefix, dev=False):
    x = json.load(open(fname, encoding='utf8'))
    print(f"Preprocessing \"{fname}\" for fine-tuning...")
    if os.path.exists(f'{out_fname_prefix}_src.txt') and os.path.exists(f'{out_fname_prefix}_tgt.txt'):
        print(f"Skipped! Fine-tuning data existed at \"{out_fname_prefix}*.txt\"")
        if NEMO_MEGATRON_CI:
            time.sleep(5)
        return
    with open(f'{out_fname_prefix}_src.txt', 'w') as f_src, \
            open(f'{out_fname_prefix}_tgt.txt', 'w') as f_tgt:
        for i in x['data']:
            title = i['title'].replace('\n', '\\n')
            for j in i['paragraphs']:
                context = j['context'].replace('\n', '\\n')
                for k in j['qas']:
                    question = k['question'].replace('\n', '\\n')
                    if len(k['answers']) > 0:
                        if dev:
                            answer = k['answers'][0]['text'].replace('\n', '\\n')
                            f_src.write(f"Title: {title} Paragraph: {context} Question: {question}\n")
                            f_tgt.write(f"{answer}\n")
                        else:
                            for a in k['answers']:
                                answer = a['text'].replace('\n', '\\n')
                                f_src.write(f"Title: {title} Paragraph: {context} Question: {question}\n")
                                f_tgt.write(f"{answer}\n")
    print(f"Completed! Fine-tuning data saved at \"{out_fname_prefix}*.txt\"")