from collections import defaultdict
from copy import copy
from typing import Dict, List, Set, Tuple

from .distance import Distance

METRIC_OPTIONS = {"f1", "span_f1", "span_distance_f1", "slu_f1"}


class ErrorMetric:
    """
    An abstract class representing a metric which accumulates TPs, FPs, and FNs.

    :param average: This determines the type of averaging performed on the data: 'micro' (calculate metrics globally by
    counting the total true positives, false negatives and false positives), 'macro' (calculate metrics for each label,
    and find their unweighted mean).
    """

    def __init__(self, average: str = "micro"):
        # These will hold per span label counts.
        self._true_positives: Dict[str, float] = defaultdict(float)
        self._false_positives: Dict[str, float] = defaultdict(float)
        self._false_negatives: Dict[str, float] = defaultdict(float)
        self._average = average

    def __call__(self, gold, prediction) -> None:
        """

        :param gold : A tensor corresponding to some gold label to evaluate against.
        :param prediction : A tensor of prediction.
        """
        raise NotImplementedError

    def get_metric(self) -> Dict[str, Tuple[float, ...]]:
        """
        Computes the metrics starting from TPs, FPs, and FNs.

        :return: A Dict per label containing a tuple for precision, recall and f1-measure. Additionally, an ``overall``
        key is included, which provides the precision, recall and f1-measure for all spans (depending on the averaging
        modality.).

        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())

        all_metrics = {}

        # Compute the precision, recall and f1 for all spans jointly.
        for tag in all_tags:
            precision, recall, f1_measure = compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )
            all_metrics[tag] = (
                precision,
                recall,
                f1_measure,
                self._true_positives[tag],
                self._false_positives[tag],
                self._false_negatives[tag],
            )
        if self._average == "macro":
            overall_precision = 0.0
            overall_recall = 0.0
            overall_f1_measure = 0.0
            for tag in all_tags:
                precision, recall, f1_measure = compute_metrics(
                    self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
                )
                overall_precision += precision
                overall_recall += recall
                overall_f1_measure += f1_measure
            precision = overall_precision / len(all_tags)
            recall = overall_recall / len(all_tags)
            f1_measure = overall_f1_measure / len(all_tags)
        else:
            precision, recall, f1_measure = compute_metrics(
                sum(self._true_positives.values()),
                sum(self._false_positives.values()),
                sum(self._false_negatives.values()),
            )
        all_metrics["overall"] = (
            precision,
            recall,
            f1_measure,
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )

        return all_metrics

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        self._true_positives: Dict[str, float] = defaultdict(float)
        self._false_positives: Dict[str, float] = defaultdict(float)
        self._false_negatives: Dict[str, float] = defaultdict(float)

    @staticmethod
    def get_instance(metric: str, average: str = "micro", distance: str = "word") -> 'ErrorMetric':
        """
        This static method allows to build a Metric object.

        :param metric: The metric to be returned.
        :param average: The average to be applied at label level.
        :param distance: When distance-based F1 is chosen, it specifies which distance is being applied.
        :return: The `Metric` object as specified by params.

        """
        assert metric in METRIC_OPTIONS, "Allowed metrics: {}".format(METRIC_OPTIONS)
        if metric == "f1":
            return FMeasure(average=average)
        if metric == "span_f1":
            return SpanFMeasure(average=average)
        if metric == "span_distance_f1":
            return SpanDistanceFMeasure(average=average, distance=distance)
        if metric == "slu_f1":
            return SLUF1(average=average)


class FMeasure(ErrorMetric):
    """
    Compute precision, recall, F-measure for each class of a general multi-class problem.
    """

    def __call__(self, gold: str, prediction: str) -> None:
        """
        This method accumulates TPs, FPs, and FNs for each label
        :param gold: The gold label.
        :param prediction: The predicted label.
        """
        if prediction == gold:
            self._true_positives[prediction] += 1
        else:
            self._false_positives[prediction] += 1
            self._false_negatives[gold] += 1


class SpanFMeasure(ErrorMetric):
    """
    Compute precision, recall, F-measure for each class of a span-based multi-class problem.
    """

    def __call__(self, gold: List[Dict[str, str]], prediction: List[Dict[str, str]]) -> None:
        """
        This method accumulates TPs, FPs, and FNs for each span label.

        :param gold: A list of gold entities, each defined by a dictionary with `type` and `filler` keys.
        :param prediction: A list of gold entities, each defined by a dictionary with `type` and `filler` keys.
        """
        gold = copy(gold)
        for entity in prediction:
            if entity in gold:
                self._true_positives[entity["type"]] += 1
                gold.remove(entity)
            else:
                self._false_positives[entity["type"]] += 1
        # These spans weren't predicted.
        for entity in gold:
            self._false_negatives[entity["type"]] += 1


class SLUF1(ErrorMetric):
    """
    The SLUF1 metric mediates between the WordF1 and CharF1, computed as the sum of the confusion matrices.
    For more information, please see `SLURP: A Spoken Language Understanding Resource Package` by Bastianelli, Vanzo,
    Swietojanski, and Rieser.
    """

    def __call__(self, results: Dict[str, Tuple[float, ...]]) -> None:
        """
        This method accumulates TPs, FPs, and FNs given the results dictionary output by another metric.

        :param results: The dictionary output by another metric.
        """
        for label in results:
            if label != "overall":
                self._true_positives[label] += results[label][3]
                self._false_positives[label] += results[label][4]
                self._false_negatives[label] += results[label][5]


class SpanDistanceFMeasure(ErrorMetric):
    """
    This metric is a generalisation of the `SpanFMeasure`, particularly suitable for measuring entity prediction scores
    in SLU tasks. A distance function is used to smooth the negative contribution of wrong transcription. In particular,
    for every label match, the lexical distance between gold and predicted fillers contributes to FPs and FNs count.
    For more information, please see `SLURP: A Spoken Language Understanding Resource Package` by Bastianelli, Vanzo,
    Swietojanski, and Rieser.

    :param average: This determines the type of averaging performed on the data: 'micro' (calculate metrics globally
    by counting the total true positives, false negatives and false positives), 'macro' (calculate metrics for each
    label, and find their unweighted mean).
    :param distance: The distance function being applied. `word` applies WER, whereas `char` applies the Levenshtein
    distance.
    """

    def __init__(self, average: str = "micro", distance: str = "word"):
        super().__init__(average=average)
        self._distance = Distance.get_instance(distance=distance)

    def __call__(self, gold: List[Dict[str, str]], prediction: List[Dict[str, str]]) -> None:
        """
        This method accumulates TPs, FPs, and FNs for each span label, taking into account the distance function being
        applied.

        :param gold: A list of gold entities, each defined by a dictionary with `type` and `filler` keys.
        :param prediction: A list of gold entities, each defined by a dictionary with `type` and `filler` keys.
        """
        gold_labels, gold_fillers = split_spans(gold)
        predicted_labels, predicted_fillers = split_spans(prediction)

        for j, pred_label in enumerate(predicted_labels):
            if pred_label in gold_labels:
                idx_to_remove, distance = self._get_lowest_distance(
                    pred_label, predicted_fillers[j], gold_labels, gold_fillers
                )
                self._true_positives[pred_label] += 1
                self._false_positives[pred_label] += distance
                self._false_negatives[pred_label] += distance
                gold_labels.pop(idx_to_remove)
                gold_fillers.pop(idx_to_remove)
            else:
                self._false_positives[pred_label] += 1
        for i, gold_label in enumerate(gold_labels):
            self._false_negatives[gold_label] += 1

    def _get_lowest_distance(
        self, target_label: str, target_span: str, gold_labels: List[str], gold_spans: List[str]
    ) -> Tuple[int, float]:
        """
        This method returns a tuple: the first element is the index of the gold entity having the lowest distance with
        the predicted one, the second element is the corresponding distance.

        :param target_label: The label of the target predicted entity.
        :param target_span: The span of the target predicted entity.
        :param gold_labels: A list of label of the gold entities. It is aligned with `gold_spans`.
        :param gold_spans: A list of span of the gold entities. It is aligned with `gold_labels`.
        :return: A tuple with index and distance of the best gold candidate.
        """
        index = 0
        lowest_distance = float("inf")
        for j, gold_label in enumerate(gold_labels):
            if target_label == gold_label:
                distance = self._distance(gold_spans[j], target_span)
                if distance < lowest_distance:
                    index = j
                    lowest_distance = distance
        return index, lowest_distance


def compute_metrics(
    true_positives: float, false_positives: float, false_negatives: float
) -> Tuple[float, float, float]:
    """
    This static method computes precision, recall and f-measure out of TPs, FPs, and FNs.

    :param true_positives: The number of true positives.
    :param false_positives: The number of false positives.
    :param false_negatives: The number of false negatives.
    :return: A tuple with precision, recall and f-measure.
    """
    if true_positives == 0.0 and false_positives == 0.0:
        precision = 0.0
    else:
        precision = float(true_positives) / float(true_positives + false_positives)
    if true_positives == 0.0 and false_negatives == 0.0:
        recall = 0.0
    else:
        recall = float(true_positives) / float(true_positives + false_negatives)
    if precision == 0.0 and recall == 0.0:
        f1_measure = 0.0
    else:
        f1_measure = 2.0 * ((precision * recall) / (precision + recall))
    return precision, recall, f1_measure


def split_spans(entities: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    """
    Split a list dictionary representing the entities into two aligned lists, containing labels and fillers,
    respectively.

    :param entities: The list of entities as dictionaries.
    :return: A tuple of lists of entities' labels and fillers.
    """
    labels = []
    fillers = []
    for entity in entities:
        if "type" not in entity or "filler" not in entity:
            continue
        labels.append(entity["type"])
        fillers.append(entity["filler"])
    return labels, fillers
