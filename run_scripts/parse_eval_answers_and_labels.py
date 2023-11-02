import json, csv, argparse

def convert_format(args):
    with open(args.input_filename) as rf:
        with open(args.input_filename + '-ang_only.csv', 'w') as wf:
            csvw = csv.writer(wf)
            for line in rf:
                ang = json.loads(line)
                csvw.writerow([ang['label'].lstrip().rstrip(), ang['pred'].lstrip().rstrip()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', type=str)
    args = parser.parse_args()

    convert_format(args)