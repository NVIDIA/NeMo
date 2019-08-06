# Taken straight from Patter https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import json
import string

from .cleaners import clean_text


def IsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def IsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize_string(s, labels, table, punctuation_to_replace,
                     **unused_kwargs):
    """
    Normalizes string. For example:
    'call me at 8:00 pm!' -> 'call me at eight zero pm'

    Args:
        s: string to normalize
        labels: labels used during model training.

    Returns:
        Normalized string
    """
    try:
        text = clean_text(s, table, punctuation_to_replace).strip()
        return text
    except BaseException:
        print("WARNING: Normalizing {} failed".format(s))
        return None


class Manifest(object):
    def __init__(self, manifest_paths, labels, max_duration=None,
                 min_duration=None, sort_by_duration=False, max_utts=0,
                 normalize=True):
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.blank_index = -1
        ids = []
        duration = 0.0
        filtered_duration = 0.0

        # If removing punctuation, make a list of punctuation to remove
        table = None
        if normalize:
            # Punctuation to remove
            punctuation = string.punctuation
            # Define punctuation that will be handled by text cleaner
            punctuation_to_replace = {
                "+": "plus",
                "&": "and",
                "%": "percent"
            }
            for char in punctuation_to_replace:
                punctuation = punctuation.replace(char, "")
            # We might also want to consider:
            # @ -> at
            # -> number, pound, hashtag
            # ~ -> tilde
            # _ -> underscore

            # If a punctuation symbol is inside our vocab, we do not remove
            # from text
            for l in labels:
                punctuation = punctuation.replace(l, "")

            # Turn all other punctuation to whitespace
            table = str.maketrans(punctuation, " " * len(punctuation))

        for manifest_path in manifest_paths:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    data = json.loads(line)
                    if min_duration is not None and data['duration'] \
                            < min_duration:
                        filtered_duration += data['duration']
                        continue
                    if max_duration is not None and data['duration'] \
                            > max_duration:
                        filtered_duration += data['duration']
                        continue

                    # Prune and normalize according to transcript
                    transcript_text = data[
                        'text'] if "text" in data else self.load_transcript(
                        data['text_filepath'])
                    if normalize:
                        transcript_text = normalize_string(
                            transcript_text,
                            labels=labels,
                            table=table,
                            punctuation_to_replace=punctuation_to_replace)
                    if not isinstance(transcript_text, str):
                        print(
                            "WARNING: Got transcript: {}. It is not a "
                            "string. Dropping data point".format(
                                transcript_text))
                        filtered_duration += data['duration']
                        continue
                    data["transcript"] = self.parse_transcript(transcript_text)
                    # data['transcript_text'] = transcript_text
                    # print(transcript_text)

                    # support files using audio_filename
                    if 'audio_filename' in data:
                        data['audio_filepath'] = data['audio_filename']
                    ids.append(data)
                    duration += data['duration']

                    if max_utts > 0 and len(ids) >= max_utts:
                        print(
                            'Stopping parsing %s as max_utts=%d' % (
                                manifest_path, max_utts))
                        break

        if sort_by_duration:
            ids = sorted(ids, key=lambda x: x['duration'])
        self._data = ids
        self._size = len(ids)
        self._duration = duration
        self._filtered_duration = filtered_duration

    def load_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        return transcript

    def parse_transcript(self, transcript):
        chars = [self.labels_map.get(x, self.blank_index)
                 for x in list(transcript)]
        transcript = list(filter(lambda x: x != self.blank_index, chars))
        return transcript

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._data)

    @property
    def duration(self):
        return self._duration

    @property
    def filtered_duration(self):
        return self._filtered_duration

    @property
    def data(self):
        return list(self._data)
