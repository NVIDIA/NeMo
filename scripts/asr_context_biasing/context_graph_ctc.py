import os
import shutil
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

# add copy rights description?


class ContextState:
    """The state in ContextGraph"""

    def __init__(
        self,
        index: int,
        is_end: bool,
        token_index: int,
    ):

        self.index = index
        self.next = {}
        self.is_end = False
        self.word = None
        self.best_token = None
        self.token_index = token_index


class ContextGraphCTC:

    def __init__(self, blank_id=1024):

        self.num_nodes = 0
        self.root = ContextState(index=self.num_nodes, is_end=False, token_index=0)
        # self.blank_token = "-"
        self.blank_token = blank_id
        

    def build(self, word_items: List[Union[str, List[List[int]]]]):

        for word_item in word_items:
            for tokens in word_item[1]:
                prev_node = self.root
                prev_token = None
                for i, token in enumerate(tokens):
                    if token not in prev_node.next:
                        self.num_nodes += 1
                        is_end = i == len(tokens) - 1
                        node = ContextState(index=self.num_nodes, is_end=is_end, token_index=i)
                        node.next[token] = node
                        prev_node.next[token] = node

                        # add blank node:
                        if prev_node is not self.root:
                            if self.blank_token in prev_node.next:
                                prev_node.next[self.blank_token].next[token] = node
                            else:
                                self.num_nodes += 1
                                blank_node = ContextState(index=self.num_nodes, is_end=False, token_index=i)    
                                blank_node.next[self.blank_token] = blank_node
                                blank_node.next[token] = node
                                prev_node.next[self.blank_token] = blank_node

                    # two consecutive equal elements:
                    if token == prev_token:
                        # already have this token in prev_node.next[balnk].next
                        if self.blank_token in prev_node.next and token in prev_node.next[self.blank_token].next:
                            prev_node = prev_node.next[self.blank_token].next[token]
                            prev_token = token
                            continue
    
                        self.num_nodes += 1
                        is_end = i == len(tokens) - 1
                        node = ContextState(index=self.num_nodes, is_end=is_end, token_index=i)
                        if self.blank_token in prev_node.next:
                            prev_node.next[self.blank_token].next[token] = node
                            node.next[token] = node
                        else:
                            node.next[token] = node 
                            # add blank node:
                            if self.blank_token in prev_node.next:
                                prev_node.next[self.blank_token].next[token] = node
                            else:
                                self.num_nodes += 1
                                blank_node = ContextState(index=self.num_nodes, is_end=False, token_index=i)    
                                blank_node.next[self.blank_token] = blank_node
                                blank_node.next[token] = node
                                prev_node.next[self.blank_token] = blank_node
                    
                    if prev_node.index != prev_node.next[token].index:
                        prev_node = prev_node.next[token]
                    else:
                        prev_node = prev_node.next[self.blank_token].next[token]
                    prev_token = token
                prev_node.is_end = True
                prev_node.word = word_item[0]
                


    def draw(
        self,
        title: Optional[str] = None,
        symbol_table: Optional[Dict[int, str]] = None,
    ) -> "Digraph":  # noqa

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
        seen.add(0)
        printed_arcs = set()

        while len(queue):
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                if node.index not in seen:
                    label = f"{node.index}"
                    if node.is_end:
                        dot.node(str(node.index), label=label, **final_state_attr)
                    else:
                        dot.node(str(node.index), label=label, **default_node_attr)
                    seen.add(node.index)
                label = str(token) if symbol_table is None else symbol_table[token]
                if node.index != current_node.index:
                    output, input, arc = str(current_node.index), str(node.index), f"{label}"
                    if (output, input, arc) not in printed_arcs:
                        dot.edge(output, input, label=arc)
                        queue.append(node)
                else:
                    output, input, arc = str(current_node.index), str(current_node.index), f"{label}"
                    if (output, input, arc) not in printed_arcs:
                        dot.edge(output, input, label=arc, color="green",)
                printed_arcs.add((output, input, arc))

        return dot