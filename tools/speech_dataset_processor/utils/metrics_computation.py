import editdistance
import difflib

sm = difflib.SequenceMatcher()


def get_cer(text, pred_text):
    char_dist = editdistance.eval(text, pred_text)
    num_chars = len(text)
    cer = round(char_dist / num_chars * 100.0, 2)

    return cer


def get_wer(text, pred_text):
    text_words = text.split()
    pred_text_words = pred_text.split()
    word_dist = editdistance.eval(text_words, pred_text_words)

    num_words = len(text_words)
    wer = round(word_dist / num_words * 100.0, 2)

    return wer


def get_charrate(text, duration):
    num_chars = len(text)
    charrate = round(num_chars / duration, 2)

    return charrate


def get_wordrate(text, duration):
    num_words = len(text.split())
    wordrate = round(num_words / duration, 2)

    return wordrate


def get_wmr(text, pred_text):
    orig = text.strip().split()
    sm.set_seqs(orig, pred_text.strip().split())
    num_matches = 0
    for m in sm.get_matching_blocks():
        for word_idx in range(m[0], m[0] + m[2]):
            num_matches += 1
    wmr = round(num_matches / len(orig) * 100.0, 2)
    return wmr
