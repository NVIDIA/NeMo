import json
import os
import argparse
import glob
import webdataset as wds


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Convert folder to webdataset script")

    # Argument parsing
    parser.add_argument("--data_path", type=str, required=False, default=None, help="Path to the folder")
    parser.add_argument("--exts", type=str, required=False, default=None, help="Examplary extensions, separated by `,`")
    args = parser.parse_args()

    folder_path = args.data_path or '.'
    files = set([f for f in glob.glob(os.path.join(folder_path, '*')) if os.path.isfile(f)])
    if not files:
        raise ValueError(f"No files in the folder: {folder_path}")

    if args.exts:
        exts = set([ext.strip('. ') for ext in args.exts.split(',')])
    else:
        # only use one extension
        exts = set([os.path.splitext(next(iter(files)))[1][1:]])

    # Paths to the dataset files
    output = os.path.join('wds')

    if not os.path.exists(output):
        os.mkdir(output)

    # Write data
    basenames = set()
    with wds.ShardWriter(os.path.join(output, 'shard-%d.tar'), maxcount=10000) as shard_writer:
        for file in files:
            basename = os.path.splitext(file)[0]
            if basename not in basenames:
                sample_files = set([basename+'.'+ext for ext in exts])
                if sample_files <= files:
                    basenames.add(basename)
                    sample = {"__key__": os.path.split(basename)[1]}
                    for s_f in sample_files:
                        with open(s_f, 'rb') as f:
                            sample[os.path.splitext(s_f)[1].strip('. ')] = f.read()
                    shard_writer.write(sample)
                    print(f"Written {basename} with keys: {sample['__key__']} to wds")

    print(f"Dataset successfully converted to wds")