import json
import os
import random

# options: "english", "nonenglish", "coding"
SEL_BLENDS = ["english", "nonenglish"]
SAMPLE_SIZE = 10000000
MAX_LENGTH = 4096

OUTPUT_FILE = f"pretraining_samples_{'_'.join(SEL_BLENDS)}.json"
DATA_FOLDER = "/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-3.5t/data/text/"
# DATA_FOLDER = "/drive3/datasets/speechlm/pretrainig/"

# NON-ENGLISH DATASETS
AR2240 = "non-english/AR_shuf.jsonl"
AZ2240 = "non-english/AZ_shuf.jsonl"
BG2240 = "non-english/BG_shuf.jsonl"
BN2240 = "non-english/BN_shuf.jsonl"
CA2240 = "non-english/CA_shuf.jsonl"
CS2240 = "non-english/CS_shuf.jsonl"
DA2240 = "non-english/DA_shuf.jsonl"
DE2240 = "non-english/DE_shuf.jsonl"
EL2240 = "non-english/EL_shuf.jsonl"
ES2240 = "non-english/ES_shuf.jsonl"
ET2240 = "non-english/ET_shuf.jsonl"
FA2240 = "non-english/FA_shuf.jsonl"
FI2240 = "non-english/FI_shuf.jsonl"
FR2240 = "non-english/FR_shuf.jsonl"
GL2240 = "non-english/GL_shuf.jsonl"
HE2240 = "non-english/HE_shuf.jsonl"
HI2240 = "non-english/HI_shuf.jsonl"
HR2240 = "non-english/HR_shuf.jsonl"
HU2240 = "non-english/HU_shuf.jsonl"
HY2240 = "non-english/HY_shuf.jsonl"
ID2240 = "non-english/ID_shuf.jsonl"
IS2240 = "non-english/IS_shuf.jsonl"
IT2240 = "non-english/IT_shuf.jsonl"
KA2240 = "non-english/KA_shuf.jsonl"
KK2240 = "non-english/KK_shuf.jsonl"
KN2240 = "non-english/KN_shuf.jsonl"
KO2240 = "non-english/KO_shuf.jsonl"
LT2240 = "non-english/LT_shuf.jsonl"
LV2240 = "non-english/LV_shuf.jsonl"
MK2240 = "non-english/MK_shuf.jsonl"
ML2240 = "non-english/ML_shuf.jsonl"
MR2240 = "non-english/MR_shuf.jsonl"
NE2240 = "non-english/NE_shuf.jsonl"
NL2240 = "non-english/NL_shuf.jsonl"
NO2240 = "non-english/NO_shuf.jsonl"
PL2240 = "non-english/PL_shuf.jsonl"
PT2240 = "non-english/PT_shuf.jsonl"
RO2240 = "non-english/RO_shuf.jsonl"
RU2240 = "non-english/RU_tdd_shuf.jsonl"
SK2240 = "non-english/SK_shuf.jsonl"
SL2240 = "non-english/SL_shuf.jsonl"
SQ2240 = "non-english/SQ_shuf.jsonl"
SR2240 = "non-english/SR_shuf.jsonl"
SV2240 = "non-english/SV_shuf.jsonl"
TA2240 = "non-english/TA_shuf.jsonl"
TE2240 = "non-english/TE_shuf.jsonl"
TR2240 = "non-english/TR_shuf.jsonl"
UK2240 = "non-english/UK_shuf.jsonl"
UR2240 = "non-english/UR_shuf.jsonl"
VI2240 = "non-english/VI_shuf.jsonl"

JAMC4 = "non-english/JA_tdd_shuf.jsonl"
ZHMC4 = "non-english/ZH_tdd_shuf.jsonl"

NMT = "non-english/nmt_tdd_shuf.jsonl"

# ENGLISH DATASETS
B3 = "english/MTNLG/Books3_shuf.jsonl"
OWT2 = "english/MTNLG/OpenWebText2_shuf.jsonl"
SE = "english/MTNLG/StackExchange_shuf.jsonl"
PMA = "english/MTNLG/PubMedAbs_shuf.jsonl"
WIK2023 = "english/MTNLG/wikipedia-05-01-2023_tdd_shuf.jsonl"
GUT = "english/MTNLG/Gutenberg_shuf.jsonl"
BC2 = "english/MTNLG/BookCorpus2_shuf.jsonl"
NIH = "english/MTNLG/NIHExporter_shuf.jsonl"
ARX2023 = "english/MTNLG/arxiv_05-05-2023_tdd_shuf.jsonl"
PMC = "english/MTNLG/pmc_tdd_shuf.jsonl"
ST = "english/MTNLG/Stories_shuf.jsonl"
BIGSC = "english/BigScience_shuf.jsonl"
REDDIT = "english/Reddit-Plus.jsonl"
CCNEWS = "english/CC-NEWS_shuf.jsonl"
PCC = "english/MTNLG/Pile-CC_shuf.jsonl"
CC201730 = "english/CC-MAIN-2017-30_tdd_shuf.jsonl"
CC201830_0 = "english/CC-MAIN-2018-30_tdd_shuf_00.jsonl"
CC201830_1 = "english/CC-MAIN-2018-30_tdd_shuf_01.jsonl"
CC201935 = "english/CC-MAIN-2019-35_shuf.jsonl"
CC202029 = "english/CC-MAIN-2020-29_tdd_shuf.jsonl"
CC202050 = "english/CC-MAIN-2020-50_shuf.jsonl"
CC202104 = "english/MTNLG/CC-2021-04_shuf.jsonl"
CC202131 = "english/CC-MAIN-2021-31_tdd_shuf.jsonl"
CC202233 = "english/CC-MAIN-2022-33_tdd_shuf.jsonl"
CC202240_0 = "english/CC-MAIN-2022-40_shuf_00.jsonl"
CC202240_1 = "english/CC-MAIN-2022-40_shuf_01.jsonl"
CC202314 = "english/CC-MAIN-2023-14_tdd_shuf.jsonl"
MC4 = "english/mc4-en_shuf.jsonl"
SEC = "english/sec_tdd_shuf.jsonl"

# CODING DATASETS
ASMB = "starcoder-repo-len-filt/assembly/assembly_repo_shuf.json"
CPLA = "starcoder-repo-len-filt/c/c_repo_shuf.json"
CSHA = "starcoder-repo-len-filt/c-sharp/c-sharp_repo_shuf.json"
CLIS = "starcoder-repo-len-filt/common-lisp/common-lisp_repo_shuf.json"
CPPP = "starcoder-repo-len-filt/cpp/cpp_repo_shuf.json"
CSSL = "starcoder-repo-len-filt/css/css_repo_shuf.json"
CUDA = "starcoder-repo-len-filt/cuda/cuda_repo_shuf.json"
DART = "starcoder-repo-len-filt/dart/dart_repo_shuf.json"
DOCK = "starcoder-repo-len-filt/dockerfile/dockerfile_repo_shuf.json"
FORT = "starcoder-repo-len-filt/fortran/fortran_repo_shuf.json"
GOPL = "starcoder-repo-len-filt/go/go_repo_shuf.json"
HASK = "starcoder-repo-len-filt/haskell/haskell_repo_shuf.json"
HTML = "starcoder-repo-len-filt/html/html_repo_shuf.json"
JAVA = "starcoder-repo-len-filt/java/java_repo_shuf.json"
JASC = "starcoder-repo-len-filt/javascript/javascript_repo_shuf.json"
JSON = "starcoder-repo-len-filt/json/json_repo_shuf.json"
JULI = "starcoder-repo-len-filt/julia/julia_repo_shuf.json"
JUPY = "starcoder-repo-len-filt/jupyter-scripts-dedup-filtered/jupyter-scripts-dedup-filtered_repo_shuf.json"
LUAL = "starcoder-repo-len-filt/lua/lua_repo_shuf.json"
MAKE = "starcoder-repo-len-filt/makefile/makefile_repo_shuf.json"
MARD = "starcoder-repo-len-filt/markdown/markdown_repo_shuf.json"
MATH = "starcoder-repo-len-filt/mathematica/mathematica_repo_shuf.json"
OMNI = "starcoder-repo-len-filt/python_merged_piiremoval.json"
PASC = "starcoder-repo-len-filt/pascal/pascal_repo_shuf.json"
PERL = "starcoder-repo-len-filt/perl/perl_repo_shuf.json"
PHPL = "starcoder-repo-len-filt/php/php_repo_shuf.json"
PYTH = "starcoder-repo-len-filt/python/python_repo_shuf.json"
RSTL = "starcoder-repo-len-filt/restructuredtext/restructuredtext_repo_shuf.json"
RUBY = "starcoder-repo-len-filt/ruby/ruby_repo_shuf.json"
RUST = "starcoder-repo-len-filt/rust/rust_repo_shuf.json"
SCAL = "starcoder-repo-len-filt/scala/scala_repo_shuf.json"
SHEL = "starcoder-repo-len-filt/shell/shell_repo_shuf.json"
SQLP = "starcoder-repo-len-filt/sql/sql_repo_shuf.json"
SWIF = "starcoder-repo-len-filt/swift_shuf.json"
SYSV = "starcoder-repo-len-filt/systemverilog/systemverilog_repo_shuf.json"
TEXP = "starcoder-repo-len-filt/tex/tex_repo_shuf.json"
TYPE = "starcoder-repo-len-filt/typescript/typescript_repo_shuf.json"
VHDL = "starcoder-repo-len-filt/vhdl/vhdl_repo_shuf.json"
VISU = "starcoder-repo-len-filt/vidual-basic/visual-basic_repo_shuf.json"
XMLL = "starcoder-repo-len-filt/xml_shuf.json"
YAML = "starcoder-repo-len-filt/yaml/yaml_repo_shuf.json"

DATA_BLEND_NONENGLISH = {
    AR2240: 0.0015,
    AZ2240: 0.00005,
    BG2240: 0.00073,
    BN2240: 0.00013,
    CA2240: 0.00036,
    CS2240: 0.00182,
    DA2240: 0.00083,
    DE2240: 0.02176,
    EL2240: 0.00166,
    ES2240: 0.02137,
    ET2240: 0.0002,
    FA2240: 0.00162,
    FI2240: 0.0008,
    FR2240: 0.01954,
    GL2240: 0.00004,
    HE2240: 0.00049,
    HI2240: 0.00047,
    HR2240: 0.0005,
    HU2240: 0.00148,
    HY2240: 0.00004,
    ID2240: 0.00322,
    IS2240: 0.00004,
    IT2240: 0.00997,
    KA2240: 0.00004,
    KK2240: 0.00004,
    KN2240: 0.00002,
    KO2240: 0.00057,
    LT2240: 0.00032,
    LV2240: 0.00014,
    MK2240: 0.00003,
    ML2240: 0.00002,
    MR2240: 0.00003,
    NE2240: 0.00003,
    NL2240: 0.00463,
    NO2240: 0.00095,
    PL2240: 0.0049,
    PT2240: 0.00689,
    RO2240: 0.00191,
    RU2240: 0.01209,
    SK2240: 0.00044,
    SL2240: 0.00023,
    SQ2240: 0.00007,
    SR2240: 0.00022,
    SV2240: 0.00158,
    TA2240: 0.0001,
    TE2240: 0.00002,
    TR2240: 0.00197,
    UK2240: 0.00117,
    UR2240: 0.00004,
    VI2240: 0.00415,
    JAMC4: 0.00827,
    ZHMC4: 0.0044,
    NMT: 0.00612,
}

DATA_BLEND_ENGLISH = {
    B3: 0.02424,
    OWT2: 0.02022,
    SE: 0.00946,
    PMA: 0.00409,
    WIK2023: 0.007,
    GUT: 0.00243,
    BC2: 0.00148,
    NIH: 0.00029,
    ARX2023: 0.02128,
    PMC: 0.01987,
    ST: 0.00301,
    BIGSC: 0.03286,
    REDDIT: 0.02644,
    CCNEWS: 0.06323,
    PCC: 0.00982,
    CC201730: 0.04649,
    CC201830_0: 0.02882,
    CC201830_1: 0.02882,
    CC201935: 0.03937,
    CC202029: 0.02946,
    CC202050: 0.03386,
    CC202104: 0.01627,
    CC202131: 0.03872,
    CC202233: 0.03828,
    CC202240_0: 0.03391,
    CC202240_1: 0.03391,
    CC202314: 0.03777,
    MC4: 0.04169,
    SEC: 0.00689,
}

DATA_BLEND_CODING = {
    ASMB: 0.0002,
    CPLA: 0.01454,
    CSHA: 0.00847,
    CLIS: 0.00005,
    CPPP: 0.01203,
    CSSL: 0.00267,
    CUDA: 0.00006,
    DART: 0.00037,
    DOCK: 0.00003,
    FORT: 0.00019,
    GOPL: 0.00582,
    HASK: 0.00022,
    HTML: 0.00577,
    JAVA: 0.0162,
    JASC: 0.01757,
    JSON: 0.00071,
    JULI: 0.00014,
    JUPY: 0.00114,
    LUAL: 0.00033,
    MAKE: 0.00015,
    MARD: 0.01666,
    MATH: 0.00014,
    OMNI: 0.00001,
    PASC: 0.0002,
    PERL: 0.00026,
    PHPL: 0.01518,
    PYTH: 0.01582,
    RSTL: 0.00032,
    RUBY: 0.00098,
    RUST: 0.00128,
    SCAL: 0.00047,
    SHEL: 0.00042,
    SQLP: 0.00292,
    SWIF: 0.00083,
    SYSV: 0.00004,
    TEXP: 0.00069,
    TYPE: 0.0051,
    VHDL: 0.00011,
    VISU: 0.00011,
    XMLL: 0.00137,
    YAML: 0.00046,
}


DATA_BLEND = {}
if "english" in SEL_BLENDS:
    DATA_BLEND.update(DATA_BLEND_ENGLISH)

if "non-english" in SEL_BLENDS:
    DATA_BLEND.update(DATA_BLEND_NONENGLISH)

if "coding" in SEL_BLENDS:
    DATA_BLEND.update(DATA_BLEND_CODING)
# A = "NIHExporter_shuf.jsonl"
# B = "NIHExporter_shuf2.jsonl"

# DATA_BLEND = {A: 0.0015, B: 0.00005}

dataset_handlers = {}

# Normalize the rates
rates_sum = 0.0
for dataset, rate in DATA_BLEND.items():
    rates_sum += rate


for dataset, rate in DATA_BLEND.items():
    DATA_BLEND[dataset] = rate / rates_sum
    dataset_path = os.path.join(DATA_FOLDER, dataset)
    dataset_handler = open(dataset_path, 'r', encoding='utf-8')
    dataset_handlers[dataset] = dataset_handler

with open(OUTPUT_FILE, 'w', encoding='utf-8') as outf:
    datasets = list(DATA_BLEND.keys())
    weights = list(DATA_BLEND.values())
    sample_idx = 0
    skipped = 0
    while sample_idx < SAMPLE_SIZE:
        if sample_idx % 10000 == 0:
            print(f"Sample_idx: {sample_idx}")
        dataset_sel = random.choices(datasets, weights=weights, k=1)[0]
        line = dataset_handlers[dataset_sel].readline()
        if line:
            sample = json.loads(line)
            # print("\n\n\n", sample.keys())
            if "text" in sample:
                entry = {'input': "", 'output': sample["text"][:MAX_LENGTH]}
                outf.write(json.dumps(entry) + '\n')
                sample_idx += 1
            else:
                # print("text field not found in ", dataset_sel)
                skipped += 1
        else:
            print(f"dataset {dataset_sel} is finished at sample_idx: {sample_idx}, skipped the selection.")
    print("Skipped: ", skipped)
