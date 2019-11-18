import argparse
import os
import csv
import json
import random

# Parsing arguments
parser = argparse.ArgumentParser(
    description='Joint intent slot filling system with pretrained BERT')
parser.add_argument("--data_dir", default='data/mturk', type=str)
parser.add_argument("--classification_data", default='classification.csv',
                    type=str)
parser.add_argument("--annotation_data", default='annotation.manifest',
                    type=str)
parser.add_argument("--anno_task_name", default='retail-data', type=str)
parser.add_argument("--num_annotators", default=3, type=str)
parser.add_argument("--dataset_name", default='mturk', type=str)
parser.add_argument("--do_lower_case", action='store_false')
parser.add_argument("--shuffle_data", action='store_false')

args = parser.parse_args()


def readCSV(file_path):
    rows = []
    with open(file_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            rows.append(row)
    return rows


def write_files(data, outfile):
    with open(f'{outfile}', 'w') as f:
        for item in data:
            item = f'{item.strip()}\n'
            f.write(item)


def analyze_inter_annotator_agreement(num_annotators, utterances,
                                      use_two_agreed=False):
    print("Agreement here")
    print(f'You had {num_annotators} annotators')

    # Assuming all the utterances are present contiguously

    size = num_annotators
    classes = []
    classDict = {}

    # TO DO - Make generalizable to n number of annotators.
    agreedall = []
    agreedtwo = []
    disagreedall = []

    intent_names = {}
    intent_count = 0

    agreedallDict = {}
    agreedallDict2 = {}

    approve_flag = False

    intent_labels = []
    intent_dict = f'{outfold}/dict.intents.csv'
    if os.path.exists(intent_dict):
        print('Printing all intent_labels')
        with open(intent_dict, 'r') as f:
            for intent_name in f.readlines():
                intent_names[intent_name.strip()] = str(intent_count)
                intent_count += 1
        print(intent_names)

    for i, utterance in enumerate(utterances[1:]):
        if size > 0:
            if utterance[1] not in classes:
                classes.append(utterance[1])
                classDict[utterance[1]] = 1
            else:
                classDict[utterance[1]] = classDict.get(utterance[1]) + 1
            size -= 1
            # print(utterance)
        if approve_flag and utterance[2] == 'x':
            if utterance[1] not in agreedallDict2:
                agreedallDict2[utterance[0]] = utterance[1]
        if size == 0:
            if len(classes) == 1:
                agreedall.append([utterance[0], classes[0]])
            elif len(classes) == 2:
                agreedtwo.append([utterance[0], classes[0], classes[1]])
            elif len(classes) == 3:
                disagreedall.append([utterance[0], classes[0],
                                    classes[1], classes[2]])
            else:
                raise ValueError(f'No of annotators is more than'
                                 f'{num_annotators}')

            # use_two_agreed - currently specific to 3 annotators only
            # TO DO : Generalize to n annotators
            if use_two_agreed and num_annotators - len(classes) + 1 == 2:
                if classDict[classes[0]] > classDict[classes[1]]:
                    agreedall.append([utterance[0], classes[0]])
                else:
                    agreedall.append([utterance[0], classes[1]])

            size = num_annotators
            classes = []
        if utterance[1] not in intent_names:
            intent_names[utterance[1]] = str(intent_count)
            intent_count += 1

    print(len(agreedall))
    print(len(agreedtwo))
    print(len(disagreedall))

    print("Full agreement")
    print(agreedall[:10])
    print("2 / 3 agreement")
    print(agreedtwo[:10])
    print("No agreement")
    print(disagreedall[:10])
    print('x eval mechanism:')
    print(len(agreedallDict2))

    if approve_flag:
        agreedallDict = agreedallDict2
    else:
        agreedallDict.update(agreedall)

    return agreedallDict, intent_names


if not os.path.exists(args.classification_data):
    raise ValueError(f'Data not found at {args.classification_data}')


utterances = []

outfold = 'mturk-processed'
os.makedirs(outfold, exist_ok=True)

utterances = readCSV(args.classification_data)


use_two_agreed = False
agreedallDict, intent_names = analyze_inter_annotator_agreement(
    args.num_annotators,
    utterances,
    use_two_agreed)


w = csv.writer(open(f'{outfold}/classification.csv', "w"))
for key, val in agreedallDict.items():
    w.writerow([key, val])

write_files(intent_names, f'{outfold}/dict.intents.csv')
