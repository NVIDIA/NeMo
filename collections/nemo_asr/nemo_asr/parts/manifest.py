# Taken straight from Patter https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import json
import string


class ManifestBase():
    def __init__(self, manifest_paths, labels, max_duration=None,
                 min_duration=None, sort_by_duration=False, max_utts=0,
                 blank_index=-1, unk_index=-1, normalize=True):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sort_by_duration = sort_by_duration
        self.max_utts = max_utts
        self.blank_index = blank_index
        self.unk_index = unk_index
        self.normalize = normalize
        self.labels_map = {label: i for i, label in enumerate(labels)}
        data = []
        duration = 0.0
        filtered_duration = 0.0

        for item in self.json_item_gen(manifest_paths):
            if min_duration and item['duration'] < min_duration:
                filtered_duration += item['duration']
                continue
            if max_duration and item['duration'] > max_duration:
                filtered_duration += item['duration']
                continue

            # load and normalize transcript text, i.e. `text`
            text = ""
            if 'text' in item:
                text = item['text']
            elif 'text_filepath' in item:
                text = self.load_transcript(item['text_filepath'])
            else:
                filtered_duration += item['duration']
                continue
            if normalize:
                text = self.normalize_text(text, labels)
            if not isinstance(text, str):
                print(
                    "WARNING: Got transcript: {}. It is not a "
                    "string. Dropping data point".format(text)
                )
                filtered_duration += item['duration']
                continue
            # item['text'] = text

            # tokenize transcript text
            item["tokens"] = self.tokenize_transcript(text)

            # support files using audio_filename
            item['audio_filepath'] = item.get('audio_filename',
                                              item['audio_filepath'])

            data.append(item)
            duration += item['duration']

            if max_utts > 0 and len(data) >= max_utts:
                print('Stop parsing due to max_utts ({})'.format(max_utts))
                break

        if sort_by_duration:
            data = sorted(data, key=lambda x: x['duration'])
        self._data = data
        self._size = len(data)
        self._duration = duration
        self._filtered_duration = filtered_duration

    def normalize_text(self, text):
        """for the base class remove surrounding whitespace only"""
        return text.strip()

    def tokenize_transcript(self, transcript):
        """tokenize transcript to convert words/characters to indices"""
        # allow for special labels such as "<NOISE>"
        special_labels = set([l for l in self.labels_map.keys() if len(l) > 1])
        tokens = []
        # split by word to find special tokens
        for i, word in enumerate(transcript.split(" ")):
            if i > 0:
                tokens.append(self.labels_map.get(" ", self.unk_index))
            if word in special_labels:
                tokens.append(self.labels_map.get(word))
                continue
            # split by character to get the rest of the tokens
            for char in word:
                tokens.append(self.labels_map.get(char, self.unk_index))
        # if unk_index == blank_index, OOV tokens are removed from transcript
        tokens = [x for x in tokens if x != self.blank_index]
        return tokens

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._data)

    @staticmethod
    def json_item_gen(manifest_paths):
        for manifest_path in manifest_paths:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    yield json.loads(line)

    @staticmethod
    def load_transcript(transcript_path):
        with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        return transcript

    @property
    def duration(self):
        return self._duration

    @property
    def filtered_duration(self):
        return self._filtered_duration

    @property
    def data(self):
        return list(self._data)


class ManifestEN(ManifestBase):
    def __init__(self, *args, **kwargs):
        from .cleaners import clean_text
        self.clean_text = clean_text
        super().__init__(*args, **kwargs)

    def normalize_text(self, text, labels):
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
        text = self.english_string_normalization(
            text,
            labels=labels,
            table=table,
            punctuation_to_replace=punctuation_to_replace,
            clean_fn=self.clean_text)

        return text

    @staticmethod
    def english_string_normalization(
            s, labels, table, punctuation_to_replace, clean_fn):
        """
        Normalizes string. For example:
        'call me at 8:00 pm!' -> 'call me at eight zero pm'

        Args:
            s: string to normalize
            labels: labels used during model training.
            table: translation table of characters to replace
            punctuation_to_replace: punctuation to be ignored in `table`
            clean_fn: function to do the cleaning

        Returns:
            Normalized string
        """
        try:
            text = clean_fn(s, table, punctuation_to_replace)
            return text
        except BaseException:
            print("WARNING: Normalizing {} failed".format(s))
            return None
