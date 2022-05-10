import argparse
from lm_eval import tasks

try:
    from nemo.utils.get_rank import is_global_rank_zero
except ModuleNotFoundError:
    print("Importing NeMo module failed, checkout the NeMo submodule")


def parse_args(parser_main):
    # parser = argparse.ArgumentParser()
    parser = parser_main.add_argument_group(title="download-tasks")
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--cache_dir", default="")
    # return parser.parse_args()
    return parser_main


def main():
    parser = argparse.ArgumentParser()
    args, unknown_args = parse_args(parser).parse_known_args()
    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    if is_global_rank_zero():
        print("***** Downloading tasks data...")
        tasks.get_task_dict(task_names, args.cache_dir)
        print("***** Tasks data downloaded.")


if __name__ == "__main__":
    main()
