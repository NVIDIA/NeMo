import logging
import random
from typing import List

import requests
import spacy
import streamlit as st
from spacy import displacy
from spacy.tokens import Doc, Span
from spacy_streamlit.util import get_html

MAX_ENTITIES = 100

DEFAULT_ENTITY_TYPES = """
beverage
restaurant
datetime
"""

DEFAULT_ENTITY_DESCRIPTIONS = """
beverage such as coffee, tea, coke
a business that provides food
a date or time
"""


def call_api(session: requests.Session, text: str, types: List[str], descriptions: List[str]):
    data = {'text': text, 'entity_types': types, 'entity_descriptions': descriptions}
    response = session.post('http://0.0.0.0:8082/label/', json=data).json()
    return response


@st.cache(allow_output_mutation=True)
def init():
    logging.info("Initializing")
    nlp = spacy.blank("en")
    session = requests.Session()
    logging.info("Done")

    random_colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(MAX_ENTITIES)]

    return session, nlp, random_colors


def main():
    session, nlp, random_colors = init()
    st.header("Zero Shot NER Demo")

    with st.form("my_form"):
        text = st.text_area(
            "* Text for zero-shot NER",
            "I would like to place an order for a large diet coke and an iced tea from McDonalds",
        )
        types = st.text_area("* Types (one entity type per line)", DEFAULT_ENTITY_TYPES.strip())
        descriptions = st.text_area(
            "* Descriptions (one entity description per line)", DEFAULT_ENTITY_DESCRIPTIONS.strip()
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            if types.strip() == '':
                types = []
                descriptions = []
            else:
                types = [_type.strip() for _type in types.strip().split("\n")]
                descriptions = [_desc.strip() for _desc in descriptions.strip().split("\n")]

            response = call_api(session, text, types, descriptions)

            types = types or response['slot_types']
            entity_id_to_string = {i: _type for i, _type in enumerate(types)}
            doc = Doc(nlp.vocab, words=response['utterance_tokens'])
            print(response['entities_dict'])
            ents = [
                Span(doc, i, j, entity_id_to_string[int(entity_id)])
                for entity_id, idx_list in response['entities_dict'].items()
                for i, j in idx_list
            ]

            doc.set_ents(ents)
            colors = random_colors[: len(types)]
            colors_dict = {_type: _color for (_type, _color) in zip(types, colors)}
            displacy_options = {'ents': types, 'colors': colors_dict}
            html = displacy.render(doc, style="ent", manual=False, options=displacy_options)
            style = "<style>mark.entity { display: inline-block }</style>"
            st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(page_title='Demo Page for Zero Shot NER')
    main()
