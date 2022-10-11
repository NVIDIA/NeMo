import json

from tqdm import tqdm

from processors.base_processor import BaseProcessor


class WriteManifest(BaseProcessor):
    def __init__(self, output_manifest_file, input_manifest_file, fields_to_save):
        self.output_manifest_file = output_manifest_file
        self.input_manifest_file = input_manifest_file
        self.fields_to_save = fields_to_save

    def process(self):
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin, open(
            self.output_manifest_file, "wt", encoding="utf8"
        ) as fout:
            for line in tqdm(fin):
                line = json.loads(line)
                new_line = {field: line[field] for field in self.fields_to_save}
                fout.write(json.dumps(new_line) + "\n")
