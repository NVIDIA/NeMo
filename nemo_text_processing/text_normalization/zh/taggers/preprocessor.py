import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst, NEMO_SIGMA
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.zh.utils import get_abs_path

class PreProcessor(GraphFst):
    '''
        Preprocessing of TN:
            1. interjections removal such as '啊, 呃'
            2. fullwidth -> halfwidth char conversion
    '''
    def __init__(self,
        remove_interjections: bool = True,
        fullwidth_to_halfwidth: bool = True,
    ):
        super().__init__(name="PreProcessor", kind="processor")  

        graph = pynini.cdrewrite('', '', '', NEMO_SIGMA)

        if remove_interjections:
            remove_interjections_graph = pynutil.delete(
                pynini.string_file(get_abs_path('data/blacklist/interjections.tsv'))
            )
            graph @= pynini.cdrewrite(remove_interjections_graph, '', '', NEMO_SIGMA)

        if fullwidth_to_halfwidth:
            fullwidth_to_halfwidth_graph = pynini.string_file(get_abs_path('data/char/fullwidth_to_halfwidth.tsv'))
            graph @= pynini.cdrewrite(fullwidth_to_halfwidth_graph, '', '', NEMO_SIGMA)

        self.fst = graph.optimize()
