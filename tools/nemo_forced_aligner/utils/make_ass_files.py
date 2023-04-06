import os
import time
from dataclasses import dataclass, field
from typing import List

PLAYERRESX = 384
PLAYERRESY = 288
MARGINL = 10
MARGINR = 10

from utils.constants import BLANK_TOKEN, SPACE_TOKEN
from utils.data_prep import Segment, Token, Utterance, Word


def seconds_to_ass_format(seconds_float):
    seconds_float = float(seconds_float)
    mm, ss_decimals = divmod(seconds_float, 60)
    hh, mm = divmod(mm, 60)

    hh = str(round(hh))
    if len(hh) == 1:
        hh = '0' + hh

    mm = str(round(mm))
    if len(mm) == 1:
        mm = '0' + mm

    ss_decimals = f"{ss_decimals:.2f}"
    if len(ss_decimals.split(".")[0]) == 1:
        ss_decimals = "0" + ss_decimals

    srt_format_time = f"{hh}:{mm}:{ss_decimals}"

    return srt_format_time


def make_ass_files(
    utt_obj, model, output_dir_root, minimum_timestamp_duration, ass_file_config,
):

    utt_obj = make_word_level_ass_file(utt_obj, model, output_dir_root, minimum_timestamp_duration, ass_file_config,)

    utt_obj = make_token_level_ass_file(utt_obj, model, output_dir_root, minimum_timestamp_duration, ass_file_config,)

    return utt_obj


def make_word_level_ass_file(
    utt_obj, model, output_dir_root, minimum_timestamp_duration, ass_file_config,
):

    default_style_dict = {
        "Name": "Default",
        "Fontname": "Arial",
        "Fontsize": str(ass_file_config.fontsize),
        "PrimaryColour": "&Hffffff",
        "SecondaryColour": "&Hffffff",
        "OutlineColour": "&H0",
        "BackColour": "&H0",
        "Bold": "0",
        "Italic": "0",
        "Underline": "0",
        "StrikeOut": "0",
        "ScaleX": "100",
        "ScaleY": "100",
        "Spacing": "0",
        "Angle": "0",
        "BorderStyle": "1",
        "Outline": "1",
        "Shadow": "0",
        "Alignment": "2",
        "MarginL": str(MARGINL),
        "MarginR": str(MARGINR),
        "MarginV": str(ass_file_config.marginv),
        "Encoding": "0",
    }

    output_dir = os.path.join(output_dir_root, "ass", "words")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{utt_obj.utt_id}.ass")

    with open(output_file, 'w') as f:
        default_style_top_line = "Format: " + ", ".join(default_style_dict.keys())
        default_style_bottom_line = "Style: " + ",".join(default_style_dict.values())

        f.write(
            (
                "[Script Info]\n"
                "ScriptType: v4.00+\n"
                f"PlayResX: {PLAYERRESX}\n"
                f"PlayResY: {PLAYERRESY}\n"
                "\n"
                "[V4+ Styles]\n"
                f"{default_style_top_line}\n"
                f"{default_style_bottom_line}\n"
                "\n"
                "[Events]\n"
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\n"
            )
        )

        # write first set of subtitles for text before speech starts to be spoken
        words_in_first_segment = []
        for segment_or_token in utt_obj.segments_and_tokens:
            if type(segment_or_token) is Segment:
                first_segment = segment_or_token

                for word_or_token in first_segment.words_and_tokens:
                    if type(word_or_token) is Word:
                        words_in_first_segment.append(word_or_token)
                break

        text_before_speech = r"{\c&c7c1c2&}" + " ".join([x.text for x in words_in_first_segment]) + r"{\r}"
        subtitle_text = (
            f"Dialogue: 0,{seconds_to_ass_format(0)},{seconds_to_ass_format(words_in_first_segment[0].t_start)},Default,,0,0,0,,"
            + text_before_speech.rstrip()
        )

        f.write(subtitle_text + '\n')

        for segment_or_token in utt_obj.segments_and_tokens:
            if type(segment_or_token) is Segment:
                segment = segment_or_token

                words_in_segment = []
                for word_or_token in segment.words_and_tokens:
                    if type(word_or_token) is Word:
                        words_in_segment.append(word_or_token)

                for word_i, word in enumerate(words_in_segment):

                    text_before = " ".join([x.text for x in words_in_segment[:word_i]])
                    if text_before != "":
                        text_before += " "
                    text_before = r"{\c&H3d2e31&}" + text_before + r"{\r}"

                    if word_i < len(words_in_segment) - 1:
                        text_after = " " + " ".join([x.text for x in words_in_segment[word_i + 1 :]])
                    else:
                        text_after = ""
                    text_after = r"{\c&c7c1c2&}" + text_after + r"{\r}"

                    aligned_text = r"{\c&H09ab39&}" + word.text + r"{\r}"
                    aligned_text_off = r"{\c&H3d2e31&}" + word.text + r"{\r}"

                    subtitle_text = (
                        f"Dialogue: 0,{seconds_to_ass_format(word.t_start)},{seconds_to_ass_format(word.t_end)},Default,,0,0,0,,"
                        + text_before
                        + aligned_text
                        + text_after.rstrip()
                    )
                    f.write(subtitle_text + '\n')

                    # add subtitles without word-highlighting for when words are not being spoken
                    if word_i < len(words_in_segment) - 1:
                        last_word_end = float(words_in_segment[word_i].t_end)
                        next_word_start = float(words_in_segment[word_i + 1].t_start)
                        if next_word_start - last_word_end > 0.001:
                            subtitle_text = (
                                f"Dialogue: 0,{seconds_to_ass_format(last_word_end)},{seconds_to_ass_format(next_word_start)},Default,,0,0,0,,"
                                + text_before
                                + aligned_text_off
                                + text_after.rstrip()
                            )
                            f.write(subtitle_text + '\n')

    utt_obj.saved_output_files[f"words_level_ass_filepath"] = output_file

    return utt_obj


def make_token_level_ass_file(
    utt_obj, model, output_dir_root, minimum_timestamp_duration, ass_file_config,
):

    default_style_dict = {
        "Name": "Default",
        "Fontname": "Arial",
        "Fontsize": str(ass_file_config.fontsize),
        "PrimaryColour": "&Hffffff",
        "SecondaryColour": "&Hffffff",
        "OutlineColour": "&H0",
        "BackColour": "&H0",
        "Bold": "0",
        "Italic": "0",
        "Underline": "0",
        "StrikeOut": "0",
        "ScaleX": "100",
        "ScaleY": "100",
        "Spacing": "0",
        "Angle": "0",
        "BorderStyle": "1",
        "Outline": "1",
        "Shadow": "0",
        "Alignment": "2",
        "MarginL": str(MARGINL),
        "MarginR": str(MARGINR),
        "MarginV": str(ass_file_config.marginv),
        "Encoding": "0",
    }

    output_dir = os.path.join(output_dir_root, "ass", "tokens")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{utt_obj.utt_id}.ass")

    with open(output_file, 'w') as f:
        default_style_top_line = "Format: " + ", ".join(default_style_dict.keys())
        default_style_bottom_line = "Style: " + ",".join(default_style_dict.values())

        f.write(
            (
                "[Script Info]\n"
                "ScriptType: v4.00+\n"
                f"PlayResX: {PLAYERRESX}\n"
                f"PlayResY: {PLAYERRESY}\n"
                "ScaledBorderAndShadow: yes\n"
                "\n"
                "[V4+ Styles]\n"
                f"{default_style_top_line}\n"
                f"{default_style_bottom_line}\n"
                "\n"
                "[Events]\n"
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\n"
            )
        )

        # write first set of subtitles for text before speech starts to be spoken
        tokens_in_first_segment = []
        for segment_or_token in utt_obj.segments_and_tokens:
            if type(segment_or_token) is Segment:
                for word_or_token in segment_or_token.words_and_tokens:
                    if type(word_or_token) is Token:
                        if word_or_token.text != BLANK_TOKEN:
                            tokens_in_first_segment.append(word_or_token)
                    else:
                        for token in word_or_token.tokens:
                            if token.text != BLANK_TOKEN:
                                tokens_in_first_segment.append(token)

                break

        for token in tokens_in_first_segment:
            token.text_cased = token.text_cased.replace(
                "▁", " "
            )  # replace underscores used in subword tokens with spaces
            token.text_cased = token.text_cased.replace(SPACE_TOKEN, " ")  # space token with actual space

        text_before_speech = r"{\c&c7c1c2&}" + "".join([x.text_cased for x in tokens_in_first_segment]) + r"{\r}"
        subtitle_text = (
            f"Dialogue: 0,{seconds_to_ass_format(0)},{seconds_to_ass_format(tokens_in_first_segment[0].t_start)},Default,,0,0,0,,"
            + text_before_speech.rstrip()
        )

        f.write(subtitle_text + '\n')

        for segment_or_token in utt_obj.segments_and_tokens:
            if type(segment_or_token) is Segment:
                segment = segment_or_token

                tokens_in_segment = []  # make list of (non-blank) tokens
                for word_or_token in segment.words_and_tokens:
                    if type(word_or_token) is Token:
                        if word_or_token.text != BLANK_TOKEN:
                            tokens_in_segment.append(word_or_token)
                    else:
                        for token in word_or_token.tokens:
                            if token.text != BLANK_TOKEN:
                                tokens_in_segment.append(token)

                for token in tokens_in_segment:
                    token.text_cased = token.text_cased.replace(
                        "▁", " "
                    )  # replace underscores used in subword tokens with spaces
                    token.text_cased = token.text_cased.replace(SPACE_TOKEN, " ")  # space token with actual space

                for token_i, token in enumerate(tokens_in_segment):

                    text_before = "".join([x.text_cased for x in tokens_in_segment[:token_i]])
                    text_before = r"{\c&H3d2e31&}" + text_before + r"{\r}"

                    if token_i < len(tokens_in_segment) - 1:
                        text_after = "".join([x.text_cased for x in tokens_in_segment[token_i + 1 :]])
                    else:
                        text_after = ""
                    text_after = r"{\c&c7c1c2&}" + text_after + r"{\r}"

                    aligned_text = r"{\c&H09ab39&}" + token.text_cased + r"{\r}"
                    aligned_text_off = r"{\c&H3d2e31&}" + token.text_cased + r"{\r}"

                    subtitle_text = (
                        f"Dialogue: 0,{seconds_to_ass_format(token.t_start)},{seconds_to_ass_format(token.t_end)},Default,,0,0,0,,"
                        + text_before
                        + aligned_text
                        + text_after.rstrip()
                    )
                    f.write(subtitle_text + '\n')

                    # add subtitles without word-highlighting for when words are not being spoken
                    if token_i < len(tokens_in_segment) - 1:
                        last_token_end = float(tokens_in_segment[token_i].t_end)
                        next_token_start = float(tokens_in_segment[token_i + 1].t_start)
                        if next_token_start - last_token_end > 0.001:
                            subtitle_text = (
                                f"Dialogue: 0,{seconds_to_ass_format(last_token_end)},{seconds_to_ass_format(next_token_start)},Default,,0,0,0,,"
                                + text_before
                                + aligned_text_off
                                + text_after.rstrip()
                            )
                            f.write(subtitle_text + '\n')

    utt_obj.saved_output_files[f"tokens_level_ass_filepath"] = output_file

    return utt_obj
