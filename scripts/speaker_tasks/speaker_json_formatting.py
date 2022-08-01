import json


def format_json(source_json_fp, out_json_fp, speaker_mapping=None):
    if speaker_mapping is None:
        speaker_mapping = {}
        spk_idx = 0

    all_records = []
    with open(source_json_fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            record = json.loads(line)
            if record['label'] in speaker_mapping:
                record['speaker'] = speaker_mapping[record['label']]
            else:
                record['speaker'] = spk_idx
                speaker_mapping[record['label']] = spk_idx
                spk_idx += 1
            record['text'] = "dummy"
            all_records.append(record)

    with open(out_json_fp, 'w') as f:
        out_str = ""
        for record in all_records:
            out_str += json.dumps(record) + '\n'
        out_str = out_str[:-1]

        f.write(out_str)

    return speaker_mapping


if __name__ == '__main__':
    speaker_mapping = format_json("/raid/datasets/train.json", "/raid/datasets/train_formatted.json")
    format_json("/raid/datasets/dev.json", "/raid/datasets/dev_formatted.json", speaker_mapping)

    # speaker_mapping = format_json("/home/shehzeenh/train.json", "/home/shehzeenh/train_formatted.json")
    # format_json("/home/shehzeenh/dev.json", "/home/shehzeenh/dev_formatted.json", speaker_mapping)
