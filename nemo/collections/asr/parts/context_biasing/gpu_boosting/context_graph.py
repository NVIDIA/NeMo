# Copyright    2023  Xiaomi Corp.        (authors: Wei Kang)
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from collections import deque
from typing import Dict, List, Optional, Tuple, Union


class ContextState:
    """The state in ContextGraph"""

    def __init__(
        self,
        id: int,
        token: int,
        token_score: float,
        node_score: float,
        output_score: float,
        is_end: bool,
        level: int,
        phrase: str = "",
        ac_threshold: float = 1.0,
    ):
        """Create a ContextState.

        Args:
          id:
            The node id, only for visualization now. A node is in [0, graph.num_nodes).
            The id of the root node is always 0.
          token:
            The token id.
          token_score:
            The bonus for each token during decoding, which will hopefully
            boost the token up to survive beam search.
          node_score:
            The accumulated bonus from root of graph to current node, it will be
            used to calculate the score for fail arc.
          output_score:
            The total scores of matched phrases, sum of the node_score of all
            the output node for current node.
          is_end:
            True if current token is the end of a context.
          level:
            The distance from current node to root.
          phrase:
            The context phrase of current state, the value is valid only when
            current state is end state (is_end == True).
          ac_threshold:
            The acoustic threshold (probability) of current context phrase, the
            value is valid only when current state is end state (is_end == True).
            Note: ac_threshold only used in keywords spotting.
        """
        self.id = id
        self.token = token
        self.token_score = token_score
        self.node_score = node_score
        self.output_score = output_score
        self.is_end = is_end
        self.level = level
        self.next = {}
        self.phrase = phrase
        self.ac_threshold = ac_threshold
        self.fail = None
        self.output = None


class ContextGraph:
    """The ContextGraph is modified from Aho-Corasick which is mainly
    a Trie with a fail arc for each node.
    See https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm for more details
    of Aho-Corasick algorithm.

    A ContextGraph contains some words / phrases that we expect to boost their
    scores during decoding. If the substring of a decoded sequence matches the word / phrase
    in the ContextGraph, we will give the decoded sequence a bonus to make it survive
    beam search.
    """

    def __init__(self, context_score: float, ac_threshold: float = 1.0):
        """Initialize a ContextGraph with the given ``context_score``.

        A root node will be created (**NOTE:** the token of root is hardcoded to -1).

        Args:
          context_score:
            The bonus score for each token(note: NOT for each word/phrase, it means longer
            word/phrase will have larger bonus score, they have to be matched though).
            Note: This is just the default score for each token, the users can manually
            specify the context_score for each word/phrase (i.e. different phrase might
            have different token score).
          ac_threshold:
            The acoustic threshold (probability) to trigger the word/phrase, this argument
            is used only when applying the graph to keywords spotting system.
        """
        self.context_score = context_score
        self.ac_threshold = ac_threshold
        self.num_nodes = 0
        self.root = ContextState(
            id=self.num_nodes,
            token=-1,
            token_score=0,
            node_score=0,
            output_score=0,
            is_end=False,
            level=0,
        )
        self.root.fail = self.root

    def _fill_fail_output(self):
        """This function fills the fail arc for each trie node, it can be computed
        in linear time by performing a breadth-first search starting from the root.
        See https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm for the
        details of the algorithm.
        """
        queue = deque()
        for token, node in self.root.next.items():
            node.fail = self.root
            queue.append(node)
        while queue:
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                fail = current_node.fail
                if token in fail.next:
                    fail = fail.next[token]
                else:
                    fail = fail.fail
                    while token not in fail.next:
                        fail = fail.fail
                        if fail.token == -1:  # root
                            break
                    if token in fail.next:
                        fail = fail.next[token]
                node.fail = fail
                # fill the output arc
                output = node.fail
                while not output.is_end:
                    output = output.fail
                    if output.token == -1:  # root
                        output = None
                        break
                node.output = output
                node.output_score += 0 if output is None else output.output_score
                queue.append(node)

    def build(
        self,
        token_ids: List[List[int]],
        phrases: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        ac_thresholds: Optional[List[float]] = None,
    ):
        """Build the ContextGraph from a list of token list.
        It first build a trie from the given token lists, then fill the fail arc
        for each trie node.

        See https://en.wikipedia.org/wiki/Trie for how to build a trie.

        Args:
          token_ids:
            The given token lists to build the ContextGraph, it is a list of
            token list, the token list contains the token ids
            for a word/phrase. The token id could be an id of a char
            (modeling with single Chinese char) or an id of a BPE
            (modeling with BPEs).
          phrases:
            The given phrases, they are the original text of the token_ids, the
            length of `phrases` MUST be equal to the length of `token_ids`.
          scores:
            The customize boosting score(token level) for each word/phrase,
            0 means using the default value (i.e. self.context_score).
            It is a list of floats, and the length of `scores` MUST be equal to
            the length of `token_ids`.
          ac_thresholds:
            The customize trigger acoustic threshold (probability) for each phrase,
            0 means using the default value (i.e. self.ac_threshold). It is
            used only when this graph applied for the keywords spotting system.
            The length of `ac_threshold` MUST be equal to the length of `token_ids`.

        Note: The phrases would have shared states, the score of the shared states is
              the MAXIMUM value among all the tokens sharing this state.
        """
        num_phrases = len(token_ids)
        if phrases is not None:
            assert len(phrases) == num_phrases, (len(phrases), num_phrases)
        if scores is not None:
            assert len(scores) == num_phrases, (len(scores), num_phrases)
        if ac_thresholds is not None:
            assert len(ac_thresholds) == num_phrases, (len(ac_thresholds), num_phrases)

        for index, tokens in enumerate(token_ids):
            phrase = "" if phrases is None else phrases[index]
            score = 0.0 if scores is None else scores[index]
            ac_threshold = 0.0 if ac_thresholds is None else ac_thresholds[index]
            node = self.root
            # If has customized score using the customized token score, otherwise
            # using the default score
            context_score = self.context_score if score == 0.0 else score
            threshold = self.ac_threshold if ac_threshold == 0.0 else ac_threshold
            for i, token in enumerate(tokens):
                node_next = {}
                if token not in node.next:
                    self.num_nodes += 1
                    is_end = i == len(tokens) - 1
                    node_score = node.node_score + context_score
                    node.next[token] = ContextState(
                        id=self.num_nodes,
                        token=token,
                        token_score=context_score,
                        node_score=node_score,
                        output_score=node_score if is_end else 0,
                        is_end=is_end,
                        level=i + 1,
                        phrase=phrase if is_end else "",
                        ac_threshold=threshold if is_end else 0.0,
                    )
                else:
                    # node exists, get the score of shared state.
                    token_score = max(context_score, node.next[token].token_score)
                    node.next[token].token_score = token_score
                    node_score = node.node_score + token_score
                    node.next[token].node_score = node_score
                    is_end = i == len(tokens) - 1 or node.next[token].is_end
                    node.next[token].output_score = node_score if is_end else 0
                    node.next[token].is_end = is_end
                    if i == len(tokens) - 1:
                        node.next[token].phrase = phrase
                        node.next[token].ac_threshold = threshold
                node = node.next[token]
        self._fill_fail_output()

    def forward_one_step(
        self, state: ContextState, token: int, strict_mode: bool = True
    ) -> Tuple[float, ContextState, ContextState]:
        """Search the graph with given state and token.

        Args:
          state:
            The given token containing trie node to start.
          token:
            The given token.
          strict_mode:
            If the `strict_mode` is True, it can match multiple phrases simultaneously,
            and will continue to match longer phrase after matching a shorter one.
            If the `strict_mode` is False, it can only match one phrase at a time,
            when it matches a phrase, then the state will fall back to root state
            (i.e. forgetting all the history state and starting a new match). If
            the matched state have multiple outputs (node.output is not None), the
            longest phrase will be return.
            For example, if the phrases are `he`, `she` and `shell`, the query is
            `like shell`, when `strict_mode` is True, the query will match `he` and
            `she` at token `e` and `shell` at token `l`, while when `strict_mode`
            if False, the query can only match `she`(`she` is longer than `he`, so
            `she` not `he`) at token `e`.
            Caution: When applying this graph for keywords spotting system, the
            `strict_mode` MUST be True.

        Returns:
          Return a tuple of boosting score for current state, next state and matched
          state (if any). Note: Only returns the matched state with longest phrase of
          current state, even if there are multiple matches phrases. If no phrase
          matched, the matched state is None.
        """
        node = None
        score = 0
        # token matched
        if token in state.next:
            node = state.next[token]
            score = node.token_score
        else:
            # token not matched
            # We will trace along the fail arc until it matches the token or reaching
            # root of the graph.
            node = state.fail
            while token not in node.next:
                node = node.fail
                if node.token == -1:  # root
                    break

            if token in node.next:
                node = node.next[token]

            # The score of the fail path
            score = node.node_score - state.node_score
        assert node is not None

        # The matched node of current step, will only return the node with
        # longest phrase if there are multiple phrases matches this step.
        # None if no matched phrase.
        matched_node = (
            node if node.is_end else (node.output if node.output is not None else None)
        )
        if not strict_mode and node.output_score != 0:
            # output_score != 0 means at least on phrase matched
            assert matched_node is not None
            output_score = (
                node.node_score
                if node.is_end
                else (
                    node.node_score if node.output is None else node.output.node_score
                )
            )
            return (score + output_score - node.node_score, self.root, matched_node)
        assert (node.output_score != 0 and matched_node is not None) or (
            node.output_score == 0 and matched_node is None
        ), (
            node.output_score,
            matched_node,
        )
        return (score + node.output_score, node, matched_node)

    def is_matched(self, state: ContextState) -> Tuple[bool, ContextState]:
        """Whether current state matches any phrase (i.e. current state is the
        end state or the output of current state is not None.

        Args:
          state:
            The given state(trie node).

        Returns:
          Return a tuple of status and matched state.
        """
        if state.is_end:
            return True, state
        else:
            if state.output is not None:
                return True, state.output
            return False, None

    def finalize(self, state: ContextState) -> Tuple[float, ContextState]:
        """When reaching the end of the decoded sequence, we need to finalize
        the matching, the purpose is to subtract the added bonus score for the
        state that is not the end of a word/phrase.

        Args:
          state:
            The given state(trie node).

        Returns:
          Return a tuple of score and next state. If state is the end of a word/phrase
          the score is zero, otherwise the score is the score of a implicit fail arc
          to root. The next state is always root.
        """
        # The score of the fail arc
        score = -state.node_score
        return (score, self.root)

    def draw(
        self,
        title: Optional[str] = None,
        filename: Optional[str] = "",
        symbol_table: Optional[Dict[int, str]] = None,
    ) -> "Digraph":  # noqa
        """Visualize a ContextGraph via graphviz.

        Render ContextGraph as an image via graphviz, and return the Digraph object;
        and optionally save to file `filename`.
        `filename` must have a suffix that graphviz understands, such as
        `pdf`, `svg` or `png`.

        Note:
          You need to install graphviz to use this function::

            pip install graphviz

        Args:
           title:
              Title to be displayed in image, e.g. 'A simple FSA example'
           filename:
              Filename to (optionally) save to, e.g. 'foo.png', 'foo.svg',
              'foo.png'  (must have a suffix that graphviz understands).
           symbol_table:
              Map the token ids to symbols.
        Returns:
          A Diagraph from grahpviz.
        """

        try:
            import graphviz
        except Exception:
            print("You cannot use `to_dot` unless the graphviz package is installed.")
            raise

        graph_attr = {
            "rankdir": "LR",
            "size": "8.5,11",
            "center": "1",
            "orientation": "Portrait",
            "ranksep": "0.4",
            "nodesep": "0.25",
        }
        if title is not None:
            graph_attr["label"] = title

        default_node_attr = {
            "shape": "circle",
            "style": "bold",
            "fontsize": "14",
        }

        final_state_attr = {
            "shape": "doublecircle",
            "style": "bold",
            "fontsize": "14",
        }

        final_state = -1
        dot = graphviz.Digraph(name="Context Graph", graph_attr=graph_attr)

        seen = set()
        queue = deque()
        queue.append(self.root)
        # root id is always 0
        dot.node("0", label="0", **default_node_attr)
        dot.edge("0", "0", color="red")
        seen.add(0)

        while len(queue):
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                if node.id not in seen:
                    node_score = f"{node.node_score:.2f}".rstrip("0").rstrip(".")
                    output_score = f"{node.output_score:.2f}".rstrip("0").rstrip(".")
                    label = f"{node.id}/({node_score}, {output_score})"
                    if node.is_end:
                        dot.node(str(node.id), label=label, **final_state_attr)
                    else:
                        dot.node(str(node.id), label=label, **default_node_attr)
                    seen.add(node.id)
                weight = f"{node.token_score:.2f}".rstrip("0").rstrip(".")
                label = str(token) if symbol_table is None else symbol_table[token]
                dot.edge(str(current_node.id), str(node.id), label=f"{label}/{weight}")
                dot.edge(
                    str(node.id),
                    str(node.fail.id),
                    color="red",
                )
                if node.output is not None:
                    dot.edge(
                        str(node.id),
                        str(node.output.id),
                        color="green",
                    )
                queue.append(node)

        if filename:
            _, extension = os.path.splitext(filename)
            if extension == "" or extension[0] != ".":
                raise ValueError(
                    "Filename needs to have a suffix like .png, .pdf, .svg: {}".format(
                        filename
                    )
                )

            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_fn = dot.render(
                    filename="temp",
                    directory=tmp_dir,
                    format=extension[1:],
                    cleanup=True,
                )

                shutil.move(temp_fn, filename)

        return dot


def _test(queries, score, strict_mode):
    contexts_str = [
        "S",
        "HE",
        "SHE",
        "SHELL",
        "HIS",
        "HERS",
        "HELLO",
        "THIS",
        "THEM",
    ]

    # test default score (1)
    contexts = []
    scores = []
    phrases = []
    for s in contexts_str:
        contexts.append([ord(x) for x in s])
        scores.append(round(score / len(s), 2))
        phrases.append(s)

    context_graph = ContextGraph(context_score=1)
    context_graph.build(token_ids=contexts, scores=scores, phrases=phrases)

    symbol_table = {}
    for contexts in contexts_str:
        for s in contexts:
            symbol_table[ord(s)] = s

    context_graph.draw(
        title="Graph for: " + " / ".join(contexts_str),
        filename=f"context_graph_{score}.pdf",
        symbol_table=symbol_table,
    )

    for query, expected_score in queries.items():
        total_scores = 0
        state = context_graph.root
        for q in query:
            score, state, phrase = context_graph.forward_one_step(
                state, ord(q), strict_mode
            )
            total_scores += score
        score, state = context_graph.finalize(state)
        assert state.token == -1, state.token
        total_scores += score
        assert round(total_scores, 2) == expected_score, (
            total_scores,
            expected_score,
            query,
        )


if __name__ == "__main__":
    # test default score
    queries = {
        "HEHERSHE": 14,  # "HE", "HE", "HERS", "S", "SHE", "HE"
        "HERSHE": 12,  # "HE", "HERS", "S", "SHE", "HE"
        "HISHE": 9,  # "HIS", "S", "SHE", "HE"
        "SHED": 6,  # "S", "SHE", "HE"
        "SHELF": 6,  # "S", "SHE", "HE"
        "HELL": 2,  # "HE"
        "HELLO": 7,  # "HE", "HELLO"
        "DHRHISQ": 4,  # "HIS", "S"
        "THEN": 2,  # "HE"
    }
    _test(queries, 0, True)

    queries = {
        "HEHERSHE": 7,  # "HE", "HE", "S", "HE"
        "HERSHE": 5,  # "HE", "S", "HE"
        "HISHE": 5,  # "HIS", "HE"
        "SHED": 3,  # "S", "HE"
        "SHELF": 3,  # "S", "HE"
        "HELL": 2,  # "HE"
        "HELLO": 2,  # "HE"
        "DHRHISQ": 3,  # "HIS"
        "THEN": 2,  # "HE"
    }
    _test(queries, 0, False)

    # test custom score
    # S : 5
    # HE : 5 (2.5 + 2.5)
    # SHE : 8.34 (5 + 1.67 + 1.67)
    # SHELL : 10.34 (5 + 1.67 + 1.67 + 1 + 1)
    # HIS : 5.84 (2.5 + 1.67 + 1.67)
    # HERS : 7.5 (2.5 + 2.5 + 1.25 + 1.25)
    # HELLO : 8 (2.5 + 2.5 + 1 + 1 + 1)
    # THIS : 5 (1.25 + 1.25 + 1.25 + 1.25)
    queries = {
        "HEHERSHE": 35.84,  # "HE", "HE", "HERS", "S", "SHE", "HE"
        "HERSHE": 30.84,  # "HE", "HERS", "S", "SHE", "HE"
        "HISHE": 24.18,  # "HIS", "S", "SHE", "HE"
        "SHED": 18.34,  # "S", "SHE", "HE"
        "SHELF": 18.34,  # "S", "SHE", "HE"
        "HELL": 5,  # "HE"
        "HELLO": 13,  # "HE", "HELLO"
        "DHRHISQ": 10.84,  # "HIS", "S"
        "THEN": 5,  # "HE"
    }

    _test(queries, 5, True)

    queries = {
        "HEHERSHE": 20,  # "HE", "HE", "S", "HE"
        "HERSHE": 15,  # "HE", "S", "HE"
        "HISHE": 10.84,  # "HIS", "HE"
        "SHED": 10,  # "S", "HE"
        "SHELF": 10,  # "S", "HE"
        "HELL": 5,  # "HE"
        "HELLO": 5,  # "HE"
        "DHRHISQ": 5.84,  # "HIS"
        "THEN": 5,  # "HE"
    }
    _test(queries, 5, False)
