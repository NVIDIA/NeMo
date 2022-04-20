import argparse
import json
import random
from tqdm import tqdm

"""
Financial Phrase Bank Dataset preprocessing script for p-tuning/prompt-tuning.

An example of the processed output written to file:
    
    {"taskname": "sentiment", "sentence": "In the Baltic countries , sales fell by 42.6 % .", "label": "negative"}
    {"taskname": "sentiment", "sentence": "Danske Bank is Denmark 's largest bank with 3.5 million customers .", "label": "neutral"}
    {"taskname": "sentiment", "sentence": "The total value of the deliveries is some EUR65m .", "label": "neutral"}
    {"taskname": "sentiment", "sentence": "Operating profit margin increased from 11.2 % to 11.7 % .", "label": "positive"}
    {"taskname": "sentiment", "sentence": "It will also strengthen Ruukki 's offshore business .", "label": "positive"}
    {"taskname": "sentiment", "sentence": "Sanoma News ' advertising sales decreased by 22 % during the year .", "label": "negative"}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/FinancialPhraseBank-v1.0")
    parser.add_argument("--file-name", type=str, default="Sentences_AllAgree.txt")
    parser.add_argument("--save-name-base", type=str, default="financial_phrase_bank")
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--train-percent", type=float, default=.8)
    args = parser.parse_args()

    data = open(f"{args.data_dir}/{args.file_name}", "r", encoding="ISO-8859-1").readlines()
    save_name_base = f"{args.data_dir}/{args.save_name_base}"

    process_data(
        data, 
        save_name_base, 
        args.train_percent, 
        args.random_seed
    )

def process_data(data, save_name_base, train_percent, random_seed):
    random.seed(random_seed)
    random.shuffle(data)

    data_total = len(data)
    train_total = int(data_total * train_percent)
    val_total = (data_total - train_total) // 2

    train_set = data[0: train_total]
    val_set = data[train_total: train_total + val_total]
    test_set = data[train_total + val_total: ]

    gen_file(train_set, save_name_base, 'train')
    gen_file(val_set, save_name_base, 'val')
    gen_file(test_set, save_name_base, 'test')

def gen_file(data, save_name_base, split_type):
    save_path = f"{save_name_base}_{split_type}.jsonl"
    print(f"Saving {split_type} split to {save_path}")

    with open(save_path, 'w') as save_file:
        for line in tqdm(data):
            example_json = {"taskname": "sentiment"}
            sent, label = line.split('@')
            sent = sent.strip()
            label = label.strip()
            example_json["sentence"] = sent
            example_json["label"] = label
            
            save_file.write(json.dumps(example_json)+'\n')

if __name__ == "__main__":
    main()