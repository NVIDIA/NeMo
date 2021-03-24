import pandas as pd
import pickle as pkl
from tqdm import tqdm
from omegaconf import OmegaConf

DATA_PATH = "MRCONSO.RRF"
HEADERS = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF']

def process_umls_training_dataset(data_path, save_name, headers):
    df = pd.read_table(data_path, names=headers, index_col=False, delimiter='|')

    with open(save_name, 'w') as outfile:
        cui = df["CUI"].iloc[0]
        names = []
        
        for idx in tqdm(range(len(df))):
            # Collect all english concept strings matching the current CUI
            if df["CUI"].iloc[idx] == cui and df["LAT"].iloc[idx] == "ENG":
                concept_string = df["STR"].iloc[idx]
                names.append(concept_string)
                
            else:
                # Pair off concept synonyms to make training set
                pairs = list(itertools.combinations(names, 2))
                
                if len(pairs) == 0:
                    # Not enough concepts gathered to make a pair
                    cui = df["CUI"].iloc[idx]
                    names = [df["STR"].iloc[idx]]
                    continue
                
                random.shuffle(pairs)
            
                # Keep up to max pairs number pairs for any one concept
                for pair in pairs[:max_pairs]:

                    # Removing leading C to convert label string to int
                    cui = cui[1:]
                    outfile.write(f'{cui}\t{pair[0]}\t{pair[1]}\n')
                
                # Switch to next concept
                cui = umls_sub["CUI"].iloc[idx]
                names = [umls_sub["STR"].iloc[idx]]

    outfile.close()


def process_umls_index_dataset(data_path, data_savename, id2string_savename, headers):
    df = pd.read_table(data_path, names=headers, index_col=False, delimiter='|')
    id2string = {}

    with open(data_savename, "w") as outfile:
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            cui = row["CUI"]
            sent = row["STR"]

            # Address incorrectly formatted data
            if "|" in cui:
                concept = cui.split("|")
                cui, sent = concept[0], concept[14]

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


process_umls_index_dataset(DATA_PATH, "umls_index_concepts.txt", "id_to_string.pkl", HEADERS)
