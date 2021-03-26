import itertools
import random
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from omegaconf import OmegaConf

DATA_PATH = "MRCONSO.RRF"
HEADERS = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF']
MAX_PAIRS = 50
TRAIN_SAVE_NAME = "umls_train_pairs.txt"
VAL_SAVE_NAME = "umls_validation_pairs.txt"
TRAIN_SPLIT = .999

def process_umls_training_dataset(data_path, train_save_name, val_save_name, headers, max_pairs, train_split):
    print("Loading training data file...")
    df = pd.read_table(data_path, names=headers, index_col=False, delimiter='|')
    train_file = open(train_save_name, 'w')
    val_file = open(val_save_name, 'w')

    cui = df["CUI"].iloc[0]
    names = []
    
    for idx in tqdm(range(len(df))):
        # Address incorrectly formatted data
        if type(df["STR"].iloc[idx]) != str or "|" in df["STR"].iloc[idx]:
            continue

        # Collect all english concept strings matching the current CUI
        if df["CUI"].iloc[idx] == cui and df["LAT"].iloc[idx] == "ENG":
            concept_string = df["STR"].iloc[idx]
            names.append(concept_string)
            
        else:
            # Pair off concept synonyms to make training and val sets
            pairs = list(itertools.combinations(names, 2))
            
            if len(pairs) == 0:
                # Not enough concepts gathered to make a pair
                cui = df["CUI"].iloc[idx]
                names = [df["STR"].iloc[idx]]
                continue
        
            # Removing leading C to convert label string to int
            cui = int(cui[1:])
            random.shuffle(pairs)

            # Keep up to max pairs number pairs for any one concept
            for pair in pairs[:max_pairs]:

                # Want concepts in train and val splits to be randomly selected and mutually exclusive 
                add_to_train = random.random()

                if add_to_train <= train_split:
                    train_file.write(f'{cui}\t{pair[0]}\t{pair[1]}\n')
                else:
                    val_file.write(f'{cui}\t{pair[0]}\t{pair[1]}\n')
            
            # Switch to next concept
            cui = df["CUI"].iloc[idx]
            names = [df["STR"].iloc[idx]]

    train_file.close()
    val_file.close()


def process_umls_index_dataset(data_path, data_savename, id2string_savename, headers):
    print("Loading index data file...")
    df = pd.read_table(data_path, names=headers, index_col=False, delimiter='|')
    id2string = {}

    with open(data_savename, "w") as outfile:
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            # Address incorrectly formatted data
            if type(row["STR"]) != str or "|" in  row["STR"]:
                continue

            cui = row["CUI"]
            sent = row["STR"]

            # Removing leading C to convert label string to int
            cui = int(cui[1:])

            # Only keeping english concepts
            if row["LAT"] == "ENG":
                outfile.write(f'{cui}\t{sent}\n')

                # Matching each cui to one canonical string represention
                if cui not in id2string and ":" not in sent:
                    id2string[cui] = sent

    outfile.close()
    pkl.dump(id2string, open(id2string_savename, "wb"))
    print("DONE!")


process_umls_training_dataset(DATA_PATH, TRAIN_SAVE_NAME, VAL_SAVE_NAME, HEADERS, MAX_PAIRS, TRAIN_SPLIT)
#process_umls_index_dataset(DATA_PATH, "umls_index_concepts.txt", "id_to_string.pkl", HEADERS)
