import argparse
import json
import random

from tqdm import tqdm


"""
Financial Phrase Bank Dataset preprocessing script for p-tuning/prompt-tuning.

An example of the processed output written to file:
    
    {
        "taskname": "squad", 
        "context": "Red is the traditional color of warning and danger. In the Middle Ages, a red flag announced that the defenders of a town or castle would fight to defend it, and a red flag hoisted by a warship meant they would show no mercy to their enemy. In Britain, in the early days of motoring, motor cars had to follow a man with a red flag who would warn horse-drawn vehicles, before the Locomotives on Highways Act 1896 abolished this law. In automobile races, the red flag is raised if there is danger to the drivers. In international football, a player who has made a serious violation of the rules is shown a red penalty card and ejected from the game.", 
        "question": "What did a red flag signal in the Middle Ages?", 
        "answer": "defenders of a town or castle would fight to defend it"
    },


"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/SQuAD")
    parser.add_argument("--file-name", type=str, default="train-v2.0.json")
    parser.add_argument("--save-name-base", type=str, default="squad")
    parser.add_argument("--make-ground-truth", action='store_true')
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--train-percent", type=float, default=0.8)
    args = parser.parse_args()

    data_dict = json.load(open(f"{args.data_dir}/{args.file_name}"))
    data = data_dict['data']
    save_name_base = f"{args.data_dir}/{args.save_name_base}"

    process_data(data, save_name_base, args.train_percent, args.random_seed, args.make_ground_truth)


def process_data(data, save_name_base, train_percent, random_seed, make_ground_truth=False):
    data = extract_questions(data)

    # Data examples are currently grouped by topic, shuffle topic groups
    random.seed(random_seed)
    random.shuffle(data)

    # Decide train/val/test splits on the topic level
    data_total = len(data)
    train_total = int(data_total * train_percent)
    val_total = (data_total - train_total) // 2

    train_set = data[0:train_total]
    val_set = data[train_total : train_total + val_total]
    test_set = data[train_total + val_total :]

    # Flatten data for each split now that topics have been confined to one split
    train_set = [question for topic in train_set for question in topic]
    val_set = [question for topic in val_set for question in topic]
    test_set = [question for topic in test_set for question in topic]

    # Shuffle train set questions
    random.shuffle(train_set)

    gen_file(train_set, save_name_base, 'train')
    gen_file(val_set, save_name_base, 'val')
    gen_file(test_set, save_name_base, 'test', make_ground_truth)


def extract_questions(data):
    processed_data = []

    # Iterate over topics, want to keep them seprate in train/val/test splits
    for question_group in data:
        processed_topic_data = []
        topic = question_group['title']
        questions = question_group['paragraphs']

        # Iterate over paragraphs related to topics
        for qa_group in questions:
            context = qa_group['context']
            qas = qa_group['qas']

            # Iterate over questions about paragraph
            for qa in qas:
                question = qa['question']

                try:
                    answer = qa['answers'][0]['text']
                except:
                    continue

                example_json = {"taskname": "squad", "context": context, "question": question, "answer": answer}
                processed_topic_data.append(example_json)
        processed_data.append(processed_topic_data)

    return processed_data


def gen_file(data, save_name_base, split_type, make_ground_truth=False):
    save_path = f"{save_name_base}_{split_type}.jsonl"
    print(f"Saving {split_type} split to {save_path}")

    with open(save_path, 'w') as save_file:
        for example_json in tqdm(data):

            # Dont want labels in the test set
            if split_type == "test" and not make_ground_truth:
                del example_json["answer"]

            save_file.write(json.dumps(example_json) + '\n')


if __name__ == "__main__":
    main()
