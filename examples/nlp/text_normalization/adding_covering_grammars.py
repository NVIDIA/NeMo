try:
    import pynini
    from pynini.lib import pynutil
    from pynini.lib import rewrite
    from nemo_text_processing.text_normalization.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.graph_utils import (
        NEMO_NON_BREAKING_SPACE,
        NEMO_SIGMA,
        SINGULAR_TO_PLURAL,
        convert_space,
        delete_space,
        insert_space,
    )
    from pynini.export import export

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

import pdb
from typing import Dict

grammar_dir = '/home/ebakhturina/TextNormalizationCoveringGrammars/src/en'

grammars = {
    'float.far': ['FLOAT'],
    'factorization.far': ['FRACTIONAL_PART_UNGROUPED', 'FRACTIONAL_PART_GROUPED', 'FRACTIONAL_PART_UNPARSED'],
    'numbers.far': ['CARDINAL_NUMBERS', 'ORDINAL_NUMBERS', 'ORDINAL_NUMBERS_UNMARKED'],
    'money.far': ['MONEY'],
    'spoken_punct.far': ['SPOKEN_PUNCT'],
    'spelled.far': ['SPELLED_NO_LETTER', 'SPELLED'],
    'number_names.far': ['CARDINAL_NUMBER_NAME', 'ORDINAL_NUMBER_NAME'],
    'numbers_plus.far': ['NUMBERS_PLUS'],
    'miscellaneous.far': ['MISCELLANEOUS'],
    'time.far': ['TIME'],
    'extra_numbers.far': ['DIGITS', 'MIXED_NUMBERS'],
    'math.far': ['ARITHMETIC'],
    'urls.far': ['URL', 'EMAIL1', 'EMAIL2', 'EMAILS'],
    'lexical_map.far': ['LEXICAL_MAP'],
    'verbalizer.far': ['VERBALIZER'],
}

grammars = {'verbalizer/verbalizer.far': ['VERBALIZER']}

# grammars = {CARDINAL_NUMBERS = Optimize[
# numbers.grm:export ORDINAL_NUMBERS_UNMARKED = Optimize[
# numbers.grm:export ORDINAL_NUMBERS_MARKED = Optimize[
# numbers.grm:export ORDINAL_NUMBERS }
fsts = {}
zero_state_fsts = {}
final_fst = None
for far, exports in grammars.items():
    far_path = f'{grammar_dir}/{far}'
    for export_ in exports:
        fst = pynini.Far(far_path, mode='r')[export_]
        if final_fst is None:
            final_fst = fst
        else:
            final_fst |= fst
        num_states = fst.num_states()
        if num_states > 0:
            fsts[far.replace('.far', '')] = fst
            # print(far.name, num_states, 'states')
        else:
            zero_state_fsts[f'{far}-{export_}'] = fst

print('zero states:', zero_state_fsts.keys())
print('final num states:', final_fst.num_states())
print(rewrite.rewrites("28", final_fst))
input = "двадцать восемь"
invert = final_fst.invert()
print(rewrite.rewrites(input, invert.optimize()))

pdb.set_trace()
print()


def _generator_main(file_name: str, graphs: Dict[str, pynini.FstLike]):
    """
    Exports graph as OpenFst finite state archive (FAR) file with given file name and rule name.

    Args:
        file_name: exported file name
        graph: Pynini WFST graph to be exported
        rule_name: rule name for graph in created FAR file

    """
    exporter = export.Exporter(file_name)
    for rule, graph in graphs.items():
        exporter[rule] = graph.optimize()
    exporter.close()
    print(f'Created {file_name}')


# _generator_main("cg_eng.far", {"VERBALIZER": final_fst})
