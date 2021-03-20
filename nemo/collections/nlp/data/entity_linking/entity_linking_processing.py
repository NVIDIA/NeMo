import pandas as pd
from tqdm import tqdm

data_path = '/home/vadams/Projects/entity-linking-research/data/umls/MRCONSO.RRF'
save_name = './umls_positive_pairs.txt'
headers = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF']

def process_umls_data(data_path, save_name, headers):
    df = pd.read_table(data_path, names=headers, index_col=False, delimiter='|')

    with open(save_name, 'w') as outfile:
        cui = df["CUI"].iloc[0]
        names = []
        
        for idx in tqdm(range(len(df))):
            # Collect all english concept strings matching the current CUI
            if df["CUI"].iloc[idx] == cui and df["LAT"].iloc[idx] == "ENG":
                concept_string = df["STR"].iloc[idx]

                # Address for misformatted data
                if "|" in concept_string:
                    concept_string = concept_string.split("|")[14]

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
                    outfile.write(f'{cui}\t{pair[0]}\t{pair[1]}\n')
                
                # Switch to next concept
                cui = umls_sub["CUI"].iloc[idx]
                names = [umls_sub["STR"].iloc[idx]]
