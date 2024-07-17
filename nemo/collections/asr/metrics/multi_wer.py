from typing import List, Union
from .wer import word_error_rate
from .eval_ner import eval_utils_ner

def get_clean_transcript(sent: str):
    clean_sent = [word for word in sent.split(' ') if not word.isupper()]
    return ' '.join(clean_sent)

def make_distinct(label_lst):
    """
    Make the label_lst distinct
    """
    tag2cnt, new_tag_lst = {}, []
    if len(label_lst) > 0:
        for tag_item in label_lst:
            _ = tag2cnt.setdefault(tag_item, 0)
            tag2cnt[tag_item] += 1
            tag, wrd = tag_item
            new_tag_lst.append((tag, wrd, tag2cnt[tag_item]))
        assert len(new_tag_lst) == len(set(new_tag_lst))
    return new_tag_lst

def get_entity_format(sents: List[str], tags: dict, score_type: str):
    def update_label_lst(lst, phrase, label):
        if label in tags['NER']:
            if score_type == "label":
                lst.append((label, "phrase"))
            else:
                lst.append((label, phrase))

    label_lst, sent_lst = [], []
    for sent in sents:
        sent_label_lst = []
        sent = sent.replace("  ", " ")
        wrd_lst = sent.split(" ")
        sent_lst.append(sent)
        phrase_lst, is_entity, num_illegal_assigments = [], False, 0
        for wrd in wrd_lst:
            if wrd in tags["NER"]:
                if is_entity:
                    phrase_lst = []
                    num_illegal_assigments += 1
                is_entity = True
                entity_tag = wrd
            elif wrd in tags["EMOTION"]:
                sent_label_lst.append((wrd, "phrase"))
            elif wrd in tags["END"]:
                if is_entity:
                    if len(phrase_lst) > 0:
                        update_label_lst(sent_label_lst, " ".join(phrase_lst), entity_tag)
                    else:
                        num_illegal_assigments += 1
                else:
                    num_illegal_assigments += 1
            else:
                if is_entity:
                    phrase_lst.append(wrd)
            label_lst.append(make_distinct(sent_label_lst))
    return label_lst, sent_lst

def multi_word_error_rate(hypotheses: List[str], references: List[str], tags: dict, score_type: str, use_cer=False) -> Union[float, float]:
    hypotheses_without_tags = [get_clean_transcript(hypothesis) for hypothesis in hypotheses]
    refs_without_tags = [get_clean_transcript(reference) for reference in references]

    wer = word_error_rate(hypotheses_without_tags, refs_without_tags, use_cer)
    
    hypo_label_list, _ = get_entity_format(hypotheses, tags, score_type)
    ref_label_list, _ = get_entity_format(references, tags, score_type)
    
    metrics = eval_utils_ner.get_ner_scores(ref_label_list, hypo_label_list)


    return wer, metrics["overall_micro"]["fscore"]

