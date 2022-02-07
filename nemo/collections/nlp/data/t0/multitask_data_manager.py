#TODO: implement a better task manager

import os

special_splits = {
    'anli': {
        'train' : ['train_r1', 'train_r2', 'train_r3'],
        'test': ['test_r1', 'test_r2', 'test_r3'],
        'validation': ['dev_r1', 'dev_r2', 'dev_r3']
    }
}

t0_traindt_names_subset = {
    # Multiple-Choice QA
    'cos_e': 'v1.11', 'cosmos_qa': None, 'commonsense_qa': None, 'dream': None,
    'qasc': None, 'quail': None, 'quarel': None, 'quartz': None,
    'sciq':None, 'social_i_qa':None, 'wiki_hop': 'original', 'wiqa': None,
    # Extractive QA
    'adversarial_qa': ['dbidaf', 'dbert', 'droberta'],
    'duorc': ['SelfRC', 'ParaphraseRC'], 'quoref': None, 'ropes': None,
    # Closed-Book QA
    'kilt_tasks':'hotpotqa', 'wiki_qa': None,
    # Structure-To-Text
    'common_gen': None, 'wiki_bio': None,
    # Sentiment
    'amazon_polarity': None, 'app_reviews': None, 'imdb': None,
    'rotten_tomatoes': None, 'yelp_review_full': None,
    # Summarization
    'cnn_dailymail': '3.0.0', 'gigaword': None, 'multi_news': None,
    'samsum': None, 'xsum': None,
    # Topic Classification
    'ag_news': None, 'dbpedia_14': None, 'trec': None,
    # Paraphrase Identification
    'glue': 'mrpc', 'glue': 'qqp', 'paws': 'labeled_final'
}
t0p_traindt_names_subset = {
    # Multiple-Choice QA
    'openbookqa': 'main', 'piqa': None, 'race': ['high', 'middle'],
    # Extractive QA
    'squad_v2': None,
    # Closed-Book QA
    'ai2_arc': ['ARC-Challenge', 'ARC-Easy'], 'trivia_qa': 'unfiltered',
    'web_questions': None
}
t0p_traindt_names_subset.update(t0_traindt_names_subset)

t0pp_traindt_names_subset = {
    'super_glue': ['copa', 'boolq', 'multirc', 'record', 'wic', 'wsc.fixed']
}

t0pp_traindt_names_subset.update(t0p_traindt_names_subset)

t0_all_evaldt_names_subset = {
    'anli': None, 'hellaswag': None,
    'super_glue': ['cb', 'copa', 'rte', 'wic', 'wsc.fixed'],
    'winogrande': 'winogrande_xl', 'story_cloze': '2016',
}


DATA_ORG = {
    "t0_train": t0_traindt_names_subset,
    "t0p_train": t0p_traindt_names_subset,
    "t0pp_train": t0pp_traindt_names_subset,
}

t0_all_names_subset = t0pp_traindt_names_subset.update(t0_all_evaldt_names_subset)

task_ids_dict = {}
id = 0
for task_name, subsets in t0_all_names_subset.items():
    if not isinstance(subsets, list):
        subsets = [subsets]
    for subset in subsets:
        subset = '' if subset is None else subset
        task_ids_dict["%s-%s" % (task_name, subset)] = id
        id += 1


def get_data_paths_and_splits(main_splits, data_dir, file_name, dt_name):
    """Handles paths to train/test/validation directories as well as special splits"""
    if not isinstance(main_splits, list):
        main_splits = [main_splits]
    if dt_name in special_splits:
        splits = [ns for s in main_splits for ns in special_splits[dt_name][s]]
        split_dirs = ['train' if 'train' in n else 'test' if 'test' in n else 'validation' for n in splits]
    else:
        splits = main_splits
        split_dirs = main_splits
    save_paths = [
        os.path.join(data_dir, split_dir, split + file_name) for split_dir, split in zip(split_dirs, splits)
    ]
    return splits, save_paths


def get_guid(task_name, subset):
    """Creates a uniques tasks id"""
    subset = '' if subset is None else subset
    task_subset_name = "%s-%s" % (task_name, subset)
    return task_ids_dict[task_subset_name]