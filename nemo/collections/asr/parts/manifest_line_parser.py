from os.path import isabs, join, split
from typing import Any, Dict


class OpenSttParser():
    @staticmethod
    def __parse(line: str, manifest_file: str) -> Dict[str, Any]:
        manifest_path, _ = split(manifest_file)
        line_list = line.split(",")
        assert len(line_list)==3, f"Incorrect fields number: {len(line_list)}"
        item = dict(
            offset=None,
            speaker=None,
            orig_sr=None,
        )

        item['audio_file'] = line_list[0]
        if not isabs(item['audio_file']):
            item['audio_file'] = join(manifest_path, item['audio_file'])

        text_file = line_list[1]
        if not isabs(text_file):
            text_file = join(manifest_path, text_file)
        with open(text_file, 'r') as f:
                item['text'] = f.read().replace('\n', '')

        item['duration'] = float(line_list[2])

        return item
