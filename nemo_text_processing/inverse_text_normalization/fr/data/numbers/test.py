import pynini
from pynini.lib import byte, pynutil

NEMO_WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", u"\u00A0").optimize()
NEMO_DIGIT = byte.DIGIT
delete_space = pynutil.delete(pynini.closure(NEMO_WHITE_SPACE))

graph_zero = pynini.string_file("zero.tsv")
graph_digit = pynini.string_file("digit.tsv")
graph_ties = pynini.string_file("ties.tsv")
graph_ties_unique = pynini.string_file("ties_unique.tsv")
graph_teen = pynini.string_file("teen.tsv")
graph_hundred = pynini.string_file("hundreds.tsv")

graph_tens_component = (
    (graph_ties + delete_space + (graph_digit | pynutil.insert("0"))) | graph_teen | graph_ties_unique
)

graph_hundred = pynini.cross(graph_hundred, "")

graph_hundred_component = pynini.union(
    (graph_digit + delete_space + graph_hundred) | pynini.cross("cent", "1"), pynutil.insert("0")
)

graph_tens_component_with_leading_zeros = graph_tens_component | (
    pynutil.insert("0") + (graph_digit | pynutil.insert("0"))
)
graph_tens_component_with_leading_zeros = delete_space + graph_tens_component_with_leading_zeros

graph_hundred_component += graph_tens_component_with_leading_zeros

graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
    pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
)

graph_thousands = pynini.union(
    graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("mille"),
    pynutil.insert("001") + pynutil.delete("mille"),  # because we say 'mille', not 'un mille'
    pynutil.insert("000", weight=0.1),
)

graph_thousands += delete_space + graph_hundred_component

graph = pynini.union(graph_thousands, graph_hundred_component, graph_zero)

graph = graph @ pynini.union(
    pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT), "0"
)

test_cases_ten = [
    'dix sept',
    'dix huit',
    'dix neuf',
    'vingt',
    'vingt et un',
    'vingt et une',
    'vingt deux',
    'vingt trois',
    'vingt quatre',
    'cinquante',
    'cinquante et un',
    'cinquante deux',
    'cinquante trois',
    'cinquante quatre',
    'soixante',
    'soixante et un',
    'soixante deux',
    'soixante quatre',
    'soixante dix',
    'soixante onze',
    'soixante dix huit',
    'quatre vingt',
    'quatre vingt un',
    'quatre vingt une',
    'quatre vingt deux',
    'quatre vingt quatre',
    'quatre vingt dix',
    'quatre vingt onze',
    'quatre vingt dix huit',
]
test_cases_hundred = [
    'cent',
    'cent un',
    'cent une',
    'cent trois',
    'cent quatre',
    'cent dix',
    'cent onze',
    'cent dix huit',
    'cent cinquante',
    'cent soixante',
    'cent soixante dix',
    'cent soixante onze',
    'cent soixante dix huit',
    'cent quatre vingt',
    'cent quatre vingt dix huit',
    'deux cents',
    'deux cent un',
    'trois cent vingt et un',
]
test_cases_thousand = [
    'mille',
    'deux mille',
    'dix mille',
    'soixante mille',
    'soixante dix mille',
    'quatre vingt mille',
    'soixante dix mille cent soixante dix',
    'quatre vingt quatre mille quatre cent quatre vingt quatre',
    'cent mille',
    'quatre cent quatre vingt dix huit mille quatre cent quatre vingt dix huit',
]

for test in test_cases_ten + test_cases_hundred + test_cases_thousand:
    paths = (test @ graph).paths()
    print(test)
    while not paths.done():
        assert paths.ostring()
        print(paths.ostring())
        paths.next()
    # print(test)
    # print((test @ graph.string()))
