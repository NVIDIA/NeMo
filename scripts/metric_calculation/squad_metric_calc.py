import argparse
import json
import re
import string
from collections import Counter


"""
Expects every line in the pred text file to have the format:

{prompt} answer: {answer}

The prediction file will be split on "answer: " when looking for the LM's predicted answer. 

"""


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--ground-truth',
        type=str,
        help="ground truth .jsonl file made from /NeMo/scripts/dataset_processing/nlp/squad/prompt_learning_squad_preprocessing.py",
    )
    parser.add_argument(
        '--preds',
        type=str,
        help="Text file with test set prompts + model predictions. Prediction file can be made by running NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_learning_eval.py",
    )

    args = parser.parse_args()

    ground_truth_file = args.ground_truth
    pred_file = args.preds

    preds = open(pred_file, encoding="utf-8").readlines()
    ground_truth = open(ground_truth_file).readlines()
    f1 = exact_match = total = 0

    for i in range(len(preds)):
        truth = json.loads(ground_truth[i])
        pred_sent = preds[i]
        pred_answer = pred_sent.split("answer:")[-1].strip()
        true_answers = truth["answer"]

        exact_match += metric_max_over_ground_truths(exact_match_score, pred_answer, true_answers)
        f1 += metric_max_over_ground_truths(f1_score, pred_answer, true_answers)
        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    print({'exact_match': exact_match, 'f1': f1, 'total': total})


if __name__ == "__main__":
    main()
