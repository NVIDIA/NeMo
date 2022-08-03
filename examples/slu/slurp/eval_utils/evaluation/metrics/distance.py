from typing import List, Union

from jiwer import wer
from textdistance.algorithms.edit_based import levenshtein

DISTANCE_OPTIONS = {"word", "char"}


class Distance:
    """
    An abstract class representing a distance metric. Every distance is normalized to be defined in [0, 1].
    """

    def __call__(self, truth: Union[str, List[str]], hypothesis: Union[str, List[str]]) -> float:
        """
        Return the distance between truth and hypothesis.

        :param truth: The ground-truth sentence as a string or list of words.
        :param hypothesis: The hypothesis sentence as a string or list of words.
        :return: The distance value between `truth` and `hypothesis`.
        """
        raise NotImplementedError

    @staticmethod
    def get_instance(distance: str) -> 'Distance':
        """
        This static method allows to build a Distance object.

        :param distance: The distance to be returned.
        :return: A `Distance` object to evaluate the distance between two texts.
        """
        assert distance in DISTANCE_OPTIONS, "Allowed distances: {}".format(DISTANCE_OPTIONS)
        if distance == "word":
            return WordDistance()
        if distance == "char":
            return CharDistance()


class WordDistance(Distance):
    """
    The Word-level distance, implemented through the Word Error Rate (WER).
    """

    def __call__(self, truth: Union[str, List[str]], hypothesis: Union[str, List[str]]) -> float:
        """
        Evaluates the word-level distance

        :param truth: The ground-truth sentence as a string or list of words.
        :param hypothesis: The hypothesis sentence as a string or list of words.
        :return: The word-level distance.
        """
        return wer(truth=truth, hypothesis=hypothesis)


class CharDistance(Distance):
    """
    The Character-level distance, implemented through the normalised Levenshtein distance.
    """

    def __call__(self, truth: Union[str, List[str]], hypothesis: Union[str, List[str]]) -> float:
        """
        Evaluates the character-level distance

        :param truth: The ground-truth sentence as a string or list of words.
        :param hypothesis: The hypothesis sentence as a string or list of words.
        :return: The character-level distance.
        """
        return levenshtein.normalized_distance(truth, hypothesis)
