import importlib
import os
import re
import sys
import warnings

parent = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent)
from . import mappings, patterns

importlib.reload(mappings)
importlib.reload(patterns)
from pathlib import Path

# from mappings import PhonemeChars, CharsPhoneme, CommonDiacritics
# from patterns import RegexPatterns
from Levenshtein import distance as levdist
from Levenshtein import editops

currentpath = Path(__file__).resolve().parent


class PersianPhonemizer:
    """
    if hamnevise=True, then for words with multiple pronounciation, multiple phoneme is generated, separated by /
    """

    def __init__(self, dictionary_path=f'{currentpath}/persian-v4.0.dict', logs=False, hamnevise=False):
        self.ZWNJ = chr(0x200C)
        self.ZIIR = chr(0x0650)
        self.TASHDID = chr(0x0651)
        self.dictionary = self.load_dictionary(dictionary_path)
        self.logs = logs
        self.patterns = patterns.RegexPatterns
        self.numberPattern = patterns.PersianNumbersPattern
        self.punctuations = r'[!،؛.؟:]'
        self.hamnevise = hamnevise

    def load_dictionary(self, path):
        dictio = {}
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if len(line.strip().split('\t')) != 2:
                    raise ValueError(f'problem with {line}')
                word, phoneme = line.strip().split('\t')
                dictio[word] = phoneme
        return dictio

    def wordcheck(self, word):
        phoneme = self.dictionary.get(word, None)
        return phoneme

    def charToUnicode(self, text):
        # convert characters of a text to unicode
        charDict = {}
        for char in text:
            charDict[char] = f'\\u{ord(char):04x}'
        return charDict

    def has_diacritics(self, word):
        return any(diacritic in word for diacritic in mappings.CommonDiacritics)

    def rmv_diacritics(self, word):
        return re.sub(f'[{"".join(mappings.CommonDiacritics)}]', '', word)

    def has_sarya(self, word):
        return "ۀ" in word

    def rmv_sarya(self, word):
        return re.sub(r'ۀ', 'ه', word)

    def rmv_punctuations(self, text):
        return re.sub(self.punctuations, '', text)

    def replace_w_with_o_in_numbers(self, text):
        match = re.search(self.numberPattern, text)
        if not match:
            return text  # Base case: no more matches, return modified text

        # Replace only the first match
        modified_text = text[: match.start()] + match.group(1) + " O " + match.group(2) + text[match.end() :]

        # Recursive call
        return self.replace_w_with_o_in_numbers(modified_text)

    def phonemize(self, text):
        text = self.rmv_punctuations(text)
        # convert WAW to O in numbers
        text = self.replace_w_with_o_in_numbers(text)

        words = text.split()
        parts = []
        for word in words:
            phoneme = self.lookup(word)
            if not self.hamnevise and "/" in phoneme:
                phoneme = phoneme.split("/")[0]
            parts.append(phoneme)
        ntext = " ".join(parts)
        phonemes = self.rephomenize(ntext)
        return phonemes.replace("'", "")

    def rephomenize(self, text):
        ptext = None
        while ptext != text:
            ptext = text
            text = self.longest(text)
        return text

    def longest(self, text):
        matches = []
        for pattern, replacement in self.patterns:
            regex = re.compile(pattern)
            for match in regex.finditer(text):
                matches.append(
                    {
                        'pattern': pattern,
                        'start': match.start(),
                        'end': match.end(),
                        'replacement': replacement,
                    }
                )

        # Early exist if there is no match
        if not matches:
            return text

        # Sort matches from **right to left** (to avoid index shifting), find the longest match and replace
        matches.sort(key=lambda x: x['start'], reverse=True)
        longest = max(matches, key=lambda x: x['end'] - x['start'])
        if self.logs:
            print(f'Matching Regex: {text} ---> {longest}')
        text = re.sub(longest['pattern'], longest['replacement'], text)
        return text

    def lookup(self, word, phoneme=""):
        # early exit if word exist in dictionary
        if word in self.dictionary:
            if self.logs:
                print(f'Dictionary: {word} --> {self.dictionary.get(word)}')
            phoneme += self.dictionary.get(word)
            return phoneme

        cword = self.rmv_sarya(word)
        cword = self.rmv_diacritics(cword)

        if self.ZWNJ in word:
            if self.logs:
                print(f'ZWNJ processing for: {word}')
            parts = word.split(self.ZWNJ)
            for part in parts:
                if part in self.dictionary:
                    phoneme += self.dictionary.get(part)
                    if self.logs:
                        print(f'ZWNJ part exists: {part} --> {self.dictionary.get(part)}')
                else:
                    if self.logs:
                        print(f'ZWNJ part does not exist, looking up: {part}')
                    phoneme = self.lookup(part, phoneme)
        elif cword in self.dictionary:
            if word == cword + self.ZIIR:
                phoneme += self.dictionary.get(cword) + "e"
                if phoneme.endswith('ee'):
                    phoneme = phoneme[:-1]
            else:
                dphoneme = self.rediacritize(word)
                phoneme += dphoneme
        else:
            # phoneme += self.splinter(word)
            phoneme += word
        return phoneme

    def rediacritize(self, word):
        # force overwrite diacritics of word into diacritics of exising
        nword = self.rmv_diacritics(word)
        nword = self.rmv_sarya(nword)

        if self.has_sarya(word) and not self.has_diacritics(word):
            sphoneme = self.dictionary.get(nword)
            dphoneme = re.sub('E', 'Y', sphoneme)
            return dphoneme

        dword = self.diacritize(nword)
        wordList = self.splitter(word)
        dwordList = self.splitter(dword)

        if len(wordList) != len(dwordList):
            if self.logs:
                print(f'Rediacritize: matching issue {word} ---> {wordList} and {dword} ---> {dwordList}')
            return word

        parts = []
        for i, ch in enumerate(wordList):
            if (self.has_diacritics(ch)) and (len(ch) >= len(dwordList[i])):
                parts.append(ch)
            else:
                parts.append(dwordList[i])
        if self.logs:
            print(f'Rediacritize: {word} ---> {"".join(parts)}')

        dphoneme = self.rephomenize("".join(parts))
        return dphoneme

    def splitter(self, word):
        # split a word based on diacritics in to parts (نِسبَت -> نِ س بَ ت)
        i, parts = 0, [""]
        while i < len(word):
            char = word[i]
            if char in mappings.CommonDiacritics:
                parts[-1] += char
            else:
                parts.append(char)
            i += 1
        if self.logs:
            print(f'Splinter: {parts}')
        return parts

    def diacritize(self, word):
        phoneme = self.dictionary.get(word, None)
        rword = ""

        if phoneme:
            wList = list(word)
            pList = list(phoneme.replace(' ', ''))
            j = 0
            while j < len(pList):
                ph = pList[j]
                if (j < len(pList) - 1) and (pList[j] == pList[j + 1]):
                    rword += mappings.CharsPhoneme.get(ph, "") + self.TASHDID
                    j += 2
                else:
                    rword += mappings.CharsPhoneme.get(ph, "")
                    j += 1
        return rword

    def splinter(self, word):
        # split words and check if each part exist in dictioanry
        parts = self.checksplits(word)
        phonemes = []
        for part in parts:
            phoneme = self.dictionary.get(part, None)
            phonemes.append(phoneme) if phoneme else phonemes.append(part)
        if self.logs:
            print(f'Splinter: {parts}')
        return "".join(phonemes)

    def checksplits(self, word):
        if word in self.dictionary:
            return [word]

        findings = []
        for i in range(1, len(word)):
            left = word[:i]
            right = word[i:]
            if left in self.dictionary and len(left) >= 4:
                rparts = self.checksplits(right)
                findings.append([left] + rparts) if rparts else findings.append([left])
        if findings:
            return max(findings, key=lambda f: len(f[0]) if f else 0)
        else:
            return [word]


if __name__ == '__main__':
    phonemizer = PersianPhonemizer(f'{currentpath}/persian-v4.0.dict', logs=False)
    texts = [
        'مبلغ هزار و سیصد پارسه',
    ]
    for text in texts:
        out = phonemizer.phonemize(text)
        print(out)
