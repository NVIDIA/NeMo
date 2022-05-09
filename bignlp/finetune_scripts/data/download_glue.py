import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile
import io

URLLIB = urllib.request

TASKS = [
    "CoLA", "SST-2", "MRPC", "QQP", "STS-B", "MNLI", 
    "QNLI", "RTE", "WNLI", "XNLI", "diagnostic",
]
TASK2PATH = {
    "CoLA": "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
    "SST-2": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    "MRPC": "https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/dev_ids.tsv",
    "QQP": "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip",
    "STS-B": "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip",
    "MNLI": "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
    "QNLI": "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
    "RTE": "https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
    "WNLI": "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip",
    "XNLI": "https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip",
    "diagnostic": "https://dl.fbaipublicfiles.com/glue/data/AX.tsv",
}

MRPC_TRAIN = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
MRPC_TEST = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"


def download_and_extract(task, data_dir):
    if os.path.exists(os.path.join(data_dir, task if task != "XNLI" else "XNLI-1.0")):
        print("Skipped downloading %s. Already exists." % task)
        return

    print("Downloading and extracting %s..." % task)
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")


def create_softlinks(tasks, data_dir):
    for task in tasks:
        src = os.path.join(data_dir, task if task != "XNLI" else "XNLI-1.0")
        dst = os.path.join(data_dir, task.lower())
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)


def format_mrpc(data_dir, path_to_data):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        try:
            mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
            mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
            URLLIB.urlretrieve(MRPC_TRAIN, mrpc_train_file)
            URLLIB.urlretrieve(MRPC_TEST, mrpc_test_file)
        except urllib.error.HTTPError:
            print("Error downloading MRPC")
            return
    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file

    with io.open(mrpc_test_file, encoding="utf-8") as data_fh, io.open(
        os.path.join(mrpc_dir, "test.tsv"), "w", encoding="utf-8"
    ) as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split("\t")
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))

    try:
        URLLIB.urlretrieve(TASK2PATH["MRPC"], os.path.join(mrpc_dir, "dev_ids.tsv"))
    except KeyError or urllib.error.HTTPError:
        print(
            "\tError downloading standard development IDs for MRPC. You will need to manually split your data."
        )
        return

    dev_ids = []
    with io.open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding="utf-8") as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split("\t"))

    with io.open(mrpc_train_file, encoding="utf-8") as data_fh, io.open(
        os.path.join(mrpc_dir, "train.tsv"), "w", encoding="utf-8"
    ) as train_fh, io.open(os.path.join(mrpc_dir, "dev.tsv"), "w", encoding="utf-8") as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split("\t")
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    print("\tCompleted!")


def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "diagnostic")):
        os.mkdir(os.path.join(data_dir, "diagnostic"))
    data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
    urllib.request.urlretrieve(TASK2PATH["diagnostic"], data_file)
    print("\tCompleted!")
    return


def get_tasks(task_names):
    task_names = task_names.split(",")
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            for task in TASKS:
                if task.lower() == task_name.lower():
                    tasks.append(task)
                    break
            else:
                raise ValueError("Task %s not found!" % task_name)

    tasks = set(tasks)
    if "XNLI" in tasks:
        tasks.add("MNLI")
    return list(tasks)


def download_glue(data_dir="glue_data", tasks="all", path_to_mrpc=None):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    tasks = get_tasks(tasks)

    for task in tasks:
        if task == "MRPC":
            format_mrpc(data_dir, path_to_mrpc)
        elif task == "diagnostic":
            download_diagnostic(data_dir)
        else:
            download_and_extract(task, data_dir)
    create_softlinks(tasks, data_dir)


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", help="directory to save data to", type=str, default="glue_data"
    )
    parser.add_argument(
        "--tasks",
        help="tasks to download data for as a comma separated string",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--path_to_mrpc",
        help="path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt",
        type=str,
        default="",
    )
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == "MRPC":
            format_mrpc(args.data_dir, args.path_to_mrpc)
        elif task == "diagnostic":
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(task, args.data_dir)
    create_softlinks(tasks, args.data_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
