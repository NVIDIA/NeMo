import os
import sys

from pynini.export import export
from tools.text_denormalization import taggers as taggers
from tools.text_denormalization import verbalizers as verbalizers


def _generator_main(file_name, graph, name):
    exporter = export.Exporter(file_name)
    exporter[name] = graph.optimize()
    exporter.close()
    print(f'Created {file_name}')


def main(output_dir):
    d = {}
    d['tokenize_and_classify'] = {'classify': taggers.tokenize_and_classify.ClassifyFst().fst}
    d['verbalize'] = {'verbalize': verbalizers.verbalize.VerbalizeFst().fst}

    for category, graphs in d.items():
        for stage, fst in graphs.items():
            out_dir = os.path.join(output_dir, stage)
            os.makedirs(out_dir, exist_ok=True)
            _generator_main(f"{out_dir}/{category}_{stage}.far", fst, category.upper())


if __name__ == '__main__':
    main(sys.argv[1])
