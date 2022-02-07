# coding=utf-8
import os
import json
from datasets import load_dataset
from argparse import ArgumentParser
from promptsource.templates import DatasetTemplates
from nemo.collections.nlp.data.t0.multitask_data_manager import (
    get_data_paths_and_splits,
    t0pp_traindt_names_subset,
    t0_all_evaldt_names_subset
)

special_dt_dir ={
    "story_cloze": "/home/jpilault/datasets/downloads/story_cloze_2016/"
}

def apply_prompts(dataset, prompts, splits, save_paths):
    for split, save_path in zip(splits, save_paths):
        counter = 0
        with open(save_path, 'w') as f:
            for example in dataset[split]:
                row = {}
                for template_name in prompts.name_to_id_mapping:
                    prompt = prompts[template_name]
                    if not prompt.metadata.original_task:
                        continue
                    result = prompt.apply(example)
                    try:
                        row[template_name] = {'input': result[0], 'output': result[1]}
                    except IndexError:
                        print(save_path)
                        print(template_name)
                        continue

                f.write(json.dumps(row))
                f.write('\n')
                counter += 1
                if counter % 100000 == 0:
                    print("{counter} applied...".format(counter=counter))

def save_raw_jsonl(dataset, prompts, splits, save_paths):
    for split, save_path in zip(splits, save_paths):
        counter = 0
        with open(save_path, 'w') as f:
            for example in dataset[split]:
                f.write(json.dumps(example))
                f.write('\n')
                counter += 1
                if counter % 100000 == 0:
                    print("{counter} applied...".format(counter=counter))

def preprocess_data(data_dict, main_splits, data_dir, save_raw):
    for dt_name in data_dict.keys():
        print(dt_name)
        subsets = data_dict[dt_name]
        if not isinstance(subsets, list):
            subsets = [subsets]
        for subset in subsets:
            print(subset)
            dataset = load_dataset(dt_name, subset, data_dir=special_dt_dir.get(dt_name, None))
            prompts = DatasetTemplates(dt_name, subset)
            file_name = "_%s_%s.jsonl" % (dt_name, "" if subset is None else subset)
            splits, save_paths = get_data_paths_and_splits(main_splits, data_dir, file_name, dt_name)
            if save_raw:
                save_raw_jsonl(dataset, prompts, splits, save_paths)
            else:
                apply_prompts(dataset, prompts, splits, save_paths)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_split", type=str, choices=['train', 'test'], help="Dataset split you want to prepare ['train', 'test']."
    )
    parser.add_argument(
        "--data_dir", type=str, default="/home/jpilault/datasets/T0_prompted", help="Parent t0 directory."
    )
    parser.add_argument(
        "--save_raw", action='store_true', default=False, help="Just save raw files to disk."
    )
    args = parser.parse_args()


    load_dataset("story_cloze", "2016", data_dir="/home/jpilault/datasets/downloads/story_cloze_2016/")

    if args.dataset_split == "train":
        data_dict = t0pp_traindt_names_subset
        main_splits = ['train']
    else:
        data_dict = t0_all_evaldt_names_subset
        main_splits = ['test', 'validation']

    preprocess_data(data_dict, main_splits, args.data_dir, args.save_raw)


if __name__ == '__main__':
    main()