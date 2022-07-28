import json
import os

import pytorch_lightning as pl
import torch
from nemo_text_processing.g2p.data.data_utils import get_wordid_to_nemo
from nemo_text_processing.g2p.models.heteronym_classification import HeteronymClassificationModel
from utils import get_metrics

from nemo.collections.tts.torch.en_utils import english_word_tokenize
from nemo.utils import logging


def correct_heteronyms(
    pretrained_heteronyms_model, manifest_with_preds, grapheme_field, batch_size: int = 32, num_workers: int = 0,
):
    # disambiguate heteronyms using IPA-classification model

    # setup GPU
    if torch.cuda.is_available():
        device = [0]  # use 0th CUDA device
        accelerator = 'gpu'
    else:
        device = 1
        accelerator = 'cpu'
    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')
    trainer = pl.Trainer(devices=device, accelerator=accelerator, logger=False, enable_checkpointing=False)

    if os.path.exists(pretrained_heteronyms_model):
        heteronyms_model = HeteronymClassificationModel.restore_from(
            pretrained_heteronyms_model, map_location=map_location
        )
    elif pretrained_heteronyms_model in HeteronymClassificationModel.get_available_model_names():
        heteronyms_model = HeteronymClassificationModel.from_pretrained(
            pretrained_heteronyms_model, map_location=map_location
        )
    else:
        raise ValueError(f'Invalid path to the pre-trained .nemo checkpoint or model name for G2PClassificationModel')

    heteronyms_model.set_trainer(trainer)
    heteronyms_model = heteronyms_model.eval()

    idx_to_wordid = {v: k for k, v in heteronyms_model.wordid_to_idx.items()}
    homograph_dict = heteronyms_model.homograph_dict
    heteronyms = homograph_dict.keys()

    grapheme_unk_token = "҂"
    ipa_unk_token = "҂"
    (
        output_manifest_with_unk,
        sentences,
        sentence_id_to_meta_info,
        start_end_indices_graphemes,
        homographs,
        heteronyms_ipa,
    ) = add_unk_token_to_manifest(
        manifest_with_preds,
        heteronyms,
        homograph_dict,
        grapheme_unk_token=grapheme_unk_token,
        ipa_unk_token=ipa_unk_token,
        grapheme_field=grapheme_field,
    )

    with torch.no_grad():
        heteronyms_preds = heteronyms_model.disambiguate(
            manifest=output_manifest_with_unk,
            grapheme_field="text_graphemes",
            batch_size=batch_size,
            num_workers=num_workers,
        )

    manifest_corrected_heteronyms = _correct_heteronym_predictions(
        manifest_with_preds, sentence_id_to_meta_info, idx_to_wordid, heteronyms_preds
    )
    get_metrics(manifest_corrected_heteronyms)


def _get_ipa_parts(words_per_segment, cur_ipa):
    try:
        current_ipa = []
        ipa_end_idx = 0
        add_previous = False
        for k, word in enumerate(words_per_segment):
            if word.isalpha():
                if add_previous:
                    import pdb

                    pdb.set_trace()
                    raise ValueError()
                if k == len(words_per_segment) - 1:
                    current_ipa.append(cur_ipa[ipa_end_idx:])
                else:
                    add_previous = True
            else:
                if add_previous:
                    new_ipa_end_idx = cur_ipa.index(word, ipa_end_idx)
                    current_ipa.append(cur_ipa[ipa_end_idx:new_ipa_end_idx])
                    ipa_end_idx = new_ipa_end_idx
                    add_previous = False

                word_len = len(word)
                current_ipa.append(cur_ipa[ipa_end_idx : ipa_end_idx + word_len])
                ipa_end_idx += word_len

        assert len(current_ipa) == len(words_per_segment)
    except Exception as e:
        return None
    return current_ipa


def add_unk_token_to_manifest(
    manifest, heteronyms, wiki_homograph_dict, grapheme_unk_token, ipa_unk_token, grapheme_field
):
    """ Replace heteronyms in graphemes (grapheme_field) and their ipa prediction () with UNKNOWN token"""
    sentences = []
    # here we're going to store sentence id (in the originla manifest file) and homograph start indices present
    # in this sentences. These values willbe later use to replace unknown tokens with classification model predictions.
    sentence_id_to_meta_info = {}
    start_end_indices_graphemes = []
    homographs = []
    heteronyms_ipa = []
    len_mismatch_skip = 0
    with open(manifest, "r", encoding="utf-8") as f_in:
        for sent_id, line in enumerate(f_in):
            line = json.loads(line)
            ipa = line["pred_text"].split()
            graphemes = line[grapheme_field]
            line["text_graphemes_original"] = line[grapheme_field]
            line["pred_text_default"] = line["pred_text"]

            if len(graphemes.split()) != len(ipa):
                logging.debug(f"len(graphemes+ != len(ipa), {line}, skipping...")
                len_mismatch_skip += 1
                continue

            graphemes_with_unk = ""
            ipa_with_unk = ""
            heteronym_present = False
            for i, segment in enumerate(graphemes.split()):
                cur_graphemes_with_unk = []
                cur_ipa_with_unk = []
                words_per_segment = [w for w, _ in english_word_tokenize(segment, lower=False)]

                # get ipa segments that match words, i.e. split punctuation marks from words
                current_ipa = _get_ipa_parts(words_per_segment, ipa[i])

                # unmatched ipa and graphemes, e.g. incorrect punctuation mark predicted
                if current_ipa is None:
                    cur_graphemes_with_unk.append(segment)
                    cur_ipa_with_unk.append(ipa[i])
                else:
                    for j, word in enumerate(words_per_segment):
                        if word.lower() in heteronyms and word.lower() in wiki_homograph_dict:
                            cur_graphemes_with_unk.append(grapheme_unk_token)
                            cur_ipa_with_unk.append(ipa_unk_token)

                            # collect data for G2P classification
                            # start and end indices for heteronyms in grapheme form
                            start_idx_grapheme = line[grapheme_field].index(word, len(" ".join(graphemes.split()[:i])))
                            end_idx_grapheme = start_idx_grapheme + len(word)
                            start_end_indices_graphemes.append([start_idx_grapheme, end_idx_grapheme])

                            # start and end indices for heteronyms in ipa form ("pred_text" field to later replace
                            # unk token with classification model prediction)
                            start_idx_ipa = line["pred_text"].index(current_ipa[j], len(" ".join(ipa[:i])))
                            end_idx_ipa = start_idx_ipa + len(current_ipa[j])

                            if sent_id not in sentence_id_to_meta_info:
                                sentence_id_to_meta_info[sent_id] = {}
                                sentence_id_to_meta_info[sent_id]["unk_indices"] = []
                                sentence_id_to_meta_info[sent_id]["default_pred"] = []

                            sentence_id_to_meta_info[sent_id]["unk_indices"].append([start_idx_ipa, end_idx_ipa])
                            sentence_id_to_meta_info[sent_id]["default_pred"].append(current_ipa[j])

                            sentences.append(graphemes)
                            homographs.append(word.lower())
                            heteronym_present = True
                        else:
                            cur_graphemes_with_unk.append(word)
                            cur_ipa_with_unk.append(current_ipa[j])

                graphemes_with_unk += " " + "".join(cur_graphemes_with_unk)

                ipa_with_unk += " " + "".join(cur_ipa_with_unk)
                cur_ipa_with_unk = []
                cur_grapheme_unk_token = []

            if heteronym_present:
                line[grapheme_field] = graphemes_with_unk.strip()

    logging.info(f"Skipped {len_mismatch_skip} due to length mismatch")

    output_manifest_with_unk = "/tmp/manifest_with_unk_token.json"
    import pdb

    pdb.set_trace()
    with open(output_manifest_with_unk, "w", encoding="utf-8") as f_out:
        for sent, homograph, ipa, start_end in zip(sentences, homographs, heteronyms_ipa, start_end_indices_graphemes):
            entry = {grapheme_field: sent, "start_end": [start_end[0], start_end[1]], "homograph_span": homograph}
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return (
        output_manifest_with_unk,
        sentences,
        sentence_id_to_meta_info,
        start_end_indices_graphemes,
        homographs,
        heteronyms_ipa,
    )


def _correct_heteronym_predictions(manifest, sentence_id_to_meta_info, target_ipa_id_to_label, heteronyms_preds):
    """
    :param manifest: path to manifest with G2P predictions in "pred_text"
    :param sentence_id_to_meta_info:
    :param target_ipa_id_to_label:
    :param heteronyms_preds: preditcions of classification G2P model
    :return:
    """
    wordid_to_nemo_cmu = get_wordid_to_nemo()
    # replace unknown token
    heteronyms_sent_id = 0
    # TODO fall back if the UNK token is not predicted -> could lead to mismatch
    manifest_corrected_heteronyms = manifest.replace(".json", "_corrected_heteronyms.json")

    with open(manifest, "r", encoding="utf-8") as f_default, open(
        manifest_corrected_heteronyms, "w", encoding="utf-8"
    ) as f_corrected:
        for sent_id, line in enumerate(f_default):
            line = json.loads(line)

            corrected_pred_text = line["pred_text"]
            offset = 0
            if sent_id in sentence_id_to_meta_info:
                for start_end_idx in sentence_id_to_meta_info[sent_id]["unk_indices"]:
                    start_idx, end_idx = start_end_idx
                    word_id = target_ipa_id_to_label[heteronyms_preds[heteronyms_sent_id]]
                    ipa_pred = wordid_to_nemo_cmu[word_id]
                    if sent_id == 0 and start_idx == 0:
                        ipa_pred = ipa_pred[:-2]
                    heteronyms_sent_id += 1
                    start_idx += offset
                    end_idx += offset
                    corrected_pred_text = corrected_pred_text[:start_idx] + ipa_pred + corrected_pred_text[end_idx:]
                    # offset to correct indices, since ipa form chosen during classification could differ in length
                    # from the predicted one
                    offset += len(ipa_pred) - (end_idx - start_idx)

            line["pred_text"] = corrected_pred_text
            f_corrected.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Saved in {manifest_corrected_heteronyms}")
    return manifest_corrected_heteronyms


if __name__ == '__main__':
    correct_heteronyms(
        "/mnt/sdb_4/g2p/chpts/homographs_classification/bert_large/G2PClassification.nemo",
        "/home/ebakhturina/CharsiuG2P/path_to_output/eval_wikihomograph.json_byt5-small.json",
        64,
        1,
    )
