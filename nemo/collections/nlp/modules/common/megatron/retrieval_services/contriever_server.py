import json
import threading

import ftfy
import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from nltk.tokenize import sent_tokenize
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.modules.common.megatron.retrieval_services.contriever import print_gpu_utilization
from nemo.collections.nlp.modules.common.megatron.retrieval_services.metrics import F1Metric
from nemo.collections.nlp.modules.common.megatron.retrieval_services.util import lock


def clean_text_with_dpr(text, dpr_tokenizer):

    embed_ids = dpr_tokenizer.encode(text, add_special_tokens=False)
    text = get_text_from_id(embed_ids, dpr_tokenizer)
    return text


def get_text_from_id(embed_ids, tokenizer):
    decode_text = tokenizer.decode(embed_ids)
    return decode_text


def format(ctx, title):
    return "title: {}, source: {}".format(title, ctx)


def evaluate_retriever(relevant_contexts, sub_paragraphs, answers, topk, dpr_tokenizer, log=False):
    ngram = 2
    threshold = 0.4

    hit = 0
    total = 0
    assert len(relevant_contexts) == len(sub_paragraphs)
    assert len(relevant_contexts) == len(answers)

    for relevant_context, answer, sub_paragraph in zip(relevant_contexts, answers, sub_paragraphs):

        gold_context = clean_text_with_dpr(sub_paragraph, dpr_tokenizer)
        candidates = []
        # print(len(relevant_context))

        found = False
        for i, ctx in enumerate(relevant_context[:topk]):
            f1 = compute_f1_score([format(*ctx)], [gold_context], n=ngram, return_recall=False)
            # candidates.append((i,f1))
            if f1 >= threshold:
                candidates.append(ctx)
                found = True
                if log:
                    print('*' * 20)
                    print(gold_context)
                    print(ctx)
                    print(answer)
                break

        if found:
            hit += 1

        total += 1
    # print(topk, hit, total)
    return hit * 1.0 / total


def compute_f1_score(predicted_answers, groundtruth_answer, n=1, return_recall=False):
    """Evaluating F1 Score"""
    # print(len(predicted_answers), len(groundtruth_answer))

    guess_list = []
    for answer in predicted_answers:
        answer = answer.strip()
        if "<|endoftext|>" in answer:
            answer = answer.replace("<|endoftext|>", "")
        guess_list.append(answer)

    answer_list = []
    for answer in groundtruth_answer:
        answer = answer.strip()
        if answer == "no_passages_used":
            answer = ""
        answer_list.append(answer)

    assert len(guess_list) == len(answer_list), "lengths of guess and answer are different!"

    precision, recall, f1 = F1Metric.compute_all_pairs(guess_list, answer_list, n)
    # print_rank_0('Method: %s; Precision: %.4f; recall: %.4f; f1: %.4f' % (\
    #     "test", precision, recall, f1))

    if return_recall:
        return recall
    return f1


def run_retriever(retriever, all_contexts, queries):
    retriever_fcn = retriever["retriever_fcn"]
    retriever_args = (
        retriever["retriever_args"] if "retriever_args" in retriever and retriever["retriever_args"] else None
    )

    if retriever_args:
        relevant_contexts = retriever_fcn(all_contexts, queries, retriever_args)
    else:
        relevant_contexts = retriever_fcn(all_contexts, queries)
    return relevant_contexts


def normalization_txt(text):
    return text.lower()


def openai_retriever(all_contexts, queries):
    """
        With openai, dot product and cosine similarity are the same, as embeddings are normalized
    """

    def openai_retriever_helper(inputs):
        import json

        import requests

        openai_api_embeddings_url = "https://api.openai.com/v1/embeddings"
        openai_api_key = "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"
        openai_api_headers = {
            'content-type': 'application/json',
            "accept": "application/json",
            "authorization": "Bearer " + openai_api_key,
        }
        openai_api_embeddings_data = {"model": "text-embedding-ada-002", "input": inputs}
        r = requests.post(
            openai_api_embeddings_url, data=json.dumps(openai_api_embeddings_data), headers=openai_api_headers
        )
        r = r.json()
        return [x["embedding"] for x in r["data"]]

    all_contexts_formatted = [normalization_txt(format(*all_context)) for all_context in all_contexts]
    all_contexts_formatted_emb = openai_retriever_helper(all_contexts_formatted)
    queries_emb = openai_retriever_helper(queries)

    all_query_scores = util.cos_sim(queries_emb, all_contexts_formatted_emb).cpu().tolist()
    all_query_docsandscores_sorted = [
        sorted(list(zip(all_contexts, query_scores)), key=lambda x: x[1], reverse=True)
        for query_scores in all_query_scores
    ]
    all_query_docs_sorted = [
        [docandscore[0] for docandscore in query_docsandscores_sorted]
        for query_docsandscores_sorted in all_query_docsandscores_sorted
    ]
    return all_query_docs_sorted


def tasb_retriever(all_contexts, queries, retriever_args):
    retrieval_method = retriever_args["retrieval_method"]
    model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
    all_contexts_formatted = [normalization_txt(format(*all_context)) for all_context in all_contexts]
    all_contexts_formatted_emb = model.encode(all_contexts_formatted)
    queries_emb = model.encode(queries)

    if retrieval_method == "DOT_PRODUCT":
        all_query_scores = util.dot_score(queries_emb, all_contexts_formatted_emb).cpu().tolist()
        all_query_docsandscores_sorted = [
            sorted(list(zip(all_contexts, query_scores)), key=lambda x: x[1], reverse=True)
            for query_scores in all_query_scores
        ]
        all_query_docs_sorted = [
            [docandscore[0] for docandscore in query_docsandscores_sorted]
            for query_docsandscores_sorted in all_query_docsandscores_sorted
        ]
        return all_query_docs_sorted
    elif retrieval_method == "COSINE_SIMILARITY":
        all_query_scores = util.cos_sim(queries_emb, all_contexts_formatted_emb).cpu().tolist()
        all_query_docsandscores_sorted = [
            sorted(list(zip(all_contexts, query_scores)), key=lambda x: x[1], reverse=True)
            for query_scores in all_query_scores
        ]
        all_query_docs_sorted = [
            [docandscore[0] for docandscore in query_docsandscores_sorted]
            for query_docsandscores_sorted in all_query_docsandscores_sorted
        ]
        return all_query_docs_sorted
    return None


def msmarcominilm_retriever(all_contexts, queries):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)
    all_query_scores = [
        list(model.predict([(query, normalization_txt(format(*context))) for context in all_contexts]))
        for query in queries
    ]
    all_query_docsandscores_sorted = [
        sorted(list(zip(all_contexts, query_scores)), key=lambda x: x[1], reverse=True)
        for query_scores in all_query_scores
    ]
    all_query_docs_sorted = [
        [docandscore[0] for docandscore in query_docsandscores_sorted]
        for query_docsandscores_sorted in all_query_docsandscores_sorted
    ]
    return all_query_docs_sorted


def msmarcominilm_reranker(all_query_contexts_ranked, queries, top_k=10):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)
    all_query_scores = [
        list(
            model.predict([(query, normalization_txt(format(*context))) for context in query_contexts_ranked[:top_k]])
        )
        for query_contexts_ranked, query in zip(all_query_contexts_ranked, queries)
    ]
    all_query_docsandscores_sorted = [
        sorted(list(zip(query_contexts_ranked[:top_k], query_scores)), key=lambda x: x[1], reverse=True)
        for query_contexts_ranked, query_scores in zip(all_query_contexts_ranked, all_query_scores)
    ]
    all_query_docs_sorted = [
        [docandscore[0] for docandscore in query_docsandscores_sorted]
        for query_docsandscores_sorted in all_query_docsandscores_sorted
    ]
    return all_query_docs_sorted


def tasb_retriever_msmarcominilm_reranker(all_contexts, queries, retriever_args):
    all_query_docs_ranked = tasb_retriever(all_contexts, queries, retriever_args)
    all_query_docs_reranked = msmarcominilm_reranker(all_query_docs_ranked, queries)
    return all_query_docs_reranked


def allminilm_retriever(all_contexts, queries):
    """
        With openai, dot product and cosine similarity are the same, as embeddings are normalized
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    all_contexts_formatted = [normalization_txt(format(*all_context)) for all_context in all_contexts]
    all_contexts_formatted_emb = model.encode(all_contexts_formatted)
    queries_emb = model.encode(queries)

    all_query_scores = util.cos_sim(queries_emb, all_contexts_formatted_emb).cpu().tolist()
    all_query_docsandscores_sorted = [
        sorted(list(zip(all_contexts, query_scores)), key=lambda x: x[1], reverse=True)
        for query_scores in all_query_scores
    ]
    all_query_docs_sorted = [
        [docandscore[0] for docandscore in query_docsandscores_sorted]
        for query_docsandscores_sorted in all_query_docsandscores_sorted
    ]
    return all_query_docs_sorted


def bm25_retriever(all_contexts, queries):

    from rank_bm25 import BM25Okapi

    tokenized_corpus = [normalization_txt(t + c).split() for t, c in all_contexts]
    bm25 = BM25Okapi(tokenized_corpus)

    ranked_docs = []
    for q in queries:
        tokenized_query = normalization_txt(q).split()
        doc_scores = bm25.get_scores(tokenized_query)
        doc_rank = (-doc_scores).argsort()
        ranked_docs.append([all_contexts[i] for i in doc_rank])
    return ranked_docs


def sbert_retriever(all_contexts, queries):
    # For every query get a sorted list of all_contexts based on similarity
    model_name = 'multi-qa-mpnet-base-dot-v1'
    # # model_name = 'all-mpnet-base-v2'
    # model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    # contriever_model, contriever_tokenizer = contriever_group()
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    print_gpu_utilization()
    bsz = 64
    to_encode = []
    all_contexts_embeddings = []
    for i in range(len(all_contexts)):
        to_encode.append(all_contexts[i][1] + ': ' + all_contexts[i][0])
        if len(to_encode) == bsz:
            all_contexts_embeddings.append(model.encode(to_encode, convert_to_tensor=True))
            print_gpu_utilization()
            to_encode = []
    if len(to_encode) > 0:
        all_contexts_embeddings.append(model.encode(to_encode, convert_to_tensor=True))

    all_contexts_embeddings = torch.cat(all_contexts_embeddings, dim=0)
    scores = torch.matmul(query_embeddings, all_contexts_embeddings.T)
    ranked_docs = []
    for i in range(len(queries)):
        doc_rank = (-scores[i]).argsort()
        ranked_docs.append([all_contexts[j] for j in doc_rank])
    return ranked_docs


def dpr_sbert_retriever(all_contexts, queries):
    # For every query get a sorted list of all_contexts based on similarity
    # model_name='multi-qa-mpnet-base-dot-v1'
    # question_model_name = 'facebook-dpr-question_encoder-single-nq-base'
    # ctx_model_name = 'facebook-dpr-ctx_encoder-single-nq-base'
    question_model_name = 'facebook-dpr-question_encoder-multiset-base'
    ctx_model_name = 'facebook-dpr-ctx_encoder-multiset-base'

    question_model = SentenceTransformer(question_model_name)
    ctx_model = SentenceTransformer(ctx_model_name)

    query_embeddings = question_model.encode(queries, convert_to_tensor=True)
    print_gpu_utilization()
    bsz = 64
    to_encode = []
    all_contexts_embeddings = []
    for i in range(len(all_contexts)):
        to_encode.append(all_contexts[i][1] + ' [SEP] ' + all_contexts[i][0])
        if len(to_encode) == bsz:
            all_contexts_embeddings.append(ctx_model.encode(to_encode, convert_to_tensor=True))
            print_gpu_utilization()
            to_encode = []
    if len(to_encode) > 0:
        all_contexts_embeddings.append(ctx_model.encode(to_encode, convert_to_tensor=True))

    all_contexts_embeddings = torch.cat(all_contexts_embeddings, dim=0)
    scores = torch.matmul(query_embeddings, all_contexts_embeddings.T)
    ranked_docs = []
    for i in range(len(queries)):
        doc_rank = (-scores[i]).argsort()
        ranked_docs.append([all_contexts[j] for j in doc_rank])
    return ranked_docs


def chunk_car_manual(
    manual,
    chunk_by_words=False,
    chunk_by_sents=True,
    chunk_size=150,
    tokenizer=None,
    chunk_by_dpr_tokenizer=False,
    overlap=False,
):
    all_chunks = []
    for idx, (text, title) in enumerate(manual):
        if chunk_by_words and not chunk_by_sents:
            num_words = chunk_size
            words = text.split()

            num_chunks = int((len(words) - 0.6) / num_words) + 1
            for i in range(num_chunks):
                chunk_words = words[i * num_words : (i + 1) * num_words]
                chunk_str = " ".join(chunk_words)
                all_chunks.append((chunk_str, title))

        elif chunk_by_sents:
            sent_list = sent_tokenize(text)
            chunk = []
            num_words = 0
            for sent in sent_list:
                num_words += len(sent.split())
                chunk.append(sent)
                if num_words >= chunk_size:
                    chunk_str = " ".join(chunk)
                    all_chunks.append((chunk_str, title))

                    if overlap:
                        num_words -= len(chunk[0].split())
                        chunk = chunk[1:]
                    else:
                        chunk = []
                        num_words = 0

            if len(chunk) > 0:
                chunk_str = " ".join(chunk)
                all_chunks.append((chunk_str, title))
                chunk = []
                num_words = 0

        elif chunk_by_dpr_tokenizer:
            dpr_tokens = 240
            raise ValueError("not implemented")

    return all_chunks


def load_car_manual(car_manual):

    with open(car_manual, 'r') as f:
        rows = json.load(f)
    return rows


def load_qa_dataset(qa_dataset):

    with open(qa_dataset, "r") as f:
        rows = json.load(f)

    queries, sub_paragraphs, answers = [], [], []
    for row in rows:
        queries.append(row["question"])
        sub_paragraphs.append(row["sub-paragraphs"])
        answers.append(row["answers"][0])
    return queries, sub_paragraphs, answers


def load_car_manual_setup(car_manual, qa_dataset):

    rows = load_car_manual(car_manual)
    manual = []
    for i, row in enumerate(rows):
        text = ftfy.fix_text(row['text'].strip())
        title = row['title'].strip()
        manual.append((text, title))
    all_contexts = chunk_car_manual(manual)

    queries, sub_paragraphs, answers = load_qa_dataset(qa_dataset)

    return all_contexts, sub_paragraphs, queries, answers


def get_dpr_group(mode="single", return_question_model=False):
    if mode == "single":
        dpr_path = "facebook/dpr-ctx_encoder-single-nq-base"
    elif mode == "multi":
        dpr_path = "facebook/dpr-ctx_encoder-multiset-base"
    else:
        raise ValueError("wrong mode for dpr")
    print(dpr_path)
    print('Building embeddings using DPR context encoder')
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_path)
    context_model = DPRContextEncoder.from_pretrained(dpr_path).cuda()
    if return_question_model:
        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_path)
        question_model = DPRQuestionEncoder.from_pretrained(dpr_path).cuda()
        return context_model, context_tokenizer, question_model, question_tokenizer
    return context_model, context_tokenizer


def get_bert_group():

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").cuda()

    return bert_model, bert_tokenizer


def main(retriever, car_manual, qa_dataset, log=False):

    topks = [1, 5, 10, 20, 50, 100, 1000]
    # topks = [1, 5]
    _, dpr_tokenizer = get_dpr_group()
    all_contexts, sub_paragraphs, queries, answers = load_car_manual_setup(car_manual, qa_dataset)
    assert len(sub_paragraphs) == len(queries)
    print("total num of all_contexts", len(all_contexts))
    relevant_contexts = run_retriever(retriever, all_contexts, queries)
    for topk in topks:
        topk_acc = evaluate_retriever(relevant_contexts, sub_paragraphs, answers, topk, dpr_tokenizer, log=log)
        # print(topk, topk_acc)
        print(topk_acc)


def save_topk(retriever, car_manual, qa_dataset, topk, output_file):

    # topks = [1, 5]
    _, dpr_tokenizer = get_dpr_group()
    all_contexts, sub_paragraphs, queries, answers = load_car_manual_setup(car_manual, qa_dataset)
    assert len(sub_paragraphs) == len(queries)
    print("total num of all_contexts", len(all_contexts))
    relevant_contexts = run_retriever(retriever, all_contexts, queries)

    save_as_json(queries, relevant_contexts, answers, topk, output_file)


def format_output(item):

    source, title = item
    return {"title": title, "text": source}


def save_as_json(queries, relevant_contexts, answers, topk, output_file):

    outputs = []
    assert len(queries) == len(relevant_contexts)
    for q, ctxs in zip(queries, relevant_contexts):
        outputs.append([q, [format_output(c) for c in ctxs[:topk]]])

    with open(output_file, "w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")


def analyze(sub_paragraphs, answers):

    assert len(sub_paragraphs) == len(answers)
    res = [
        compute_f1_score([sub_paragraph], [answer], n=2, return_recall=True)
        for sub_paragraph, answer in zip(sub_paragraphs, answers)
    ]

    print(min([i for i in res if i > 0.1]))
    print(res)

    all_lens = [len(para.split()) for para in sub_paragraphs]
    print(max(all_lens), min(all_lens))
    print(sorted(all_lens))


def save_all(retriever_name):

    topk = 10
    retriever = all_retrievers[retriever_name]

    for split in ["train", "valid", "test"]:
        car_manual = "../data/landrover/landrover_manual.json"
        qa_dataset = "../data/landrover/{}.json".format(split)
        output_file = "../data/landrover/{}_retriever_top{}_{}.json".format(retriever_name, topk, split)
        save_topk(retriever, car_manual, qa_dataset, topk, output_file)

    for split in ["train", "valid", "test"]:
        car_manual = "../data/benz/mb-manual.json"
        qa_dataset = "../data/benz/{}.json".format(split)
        output_file = "../data/benz/{}_retriever_top{}_{}.json".format(retriever_name, topk, split)
        save_topk(retriever, car_manual, qa_dataset, topk, output_file)


def tasb_retriever_msmarcominilm_reranker(all_contexts, queries, retriever_args):
    all_query_docs_ranked = tasb_retriever(all_contexts, queries, retriever_args)
    all_query_docs_reranked = msmarcominilm_reranker(all_query_docs_ranked, queries)
    return all_query_docs_reranked


def get_relevant_context(questions, all_contexts, all_context_emb, neighbors, models):
    buffer_size = 20
    all_contexts_formatted_emb = all_context_emb
    queries_emb = models[0].encode(questions)

    all_query_scores = util.dot_score(queries_emb, all_contexts_formatted_emb).cpu().tolist()
    all_query_docsandscores_sorted = [
        sorted(list(zip(all_contexts, query_scores)), key=lambda x: x[1], reverse=True)
        for query_scores in all_query_scores
    ]
    all_query_docs_sorted = [
        [docandscore[0] for docandscore in query_docsandscores_sorted]
        for query_docsandscores_sorted in all_query_docsandscores_sorted
    ]

    if isinstance(models[1], tuple):
        rank_model, rank_model_tokenizer = models[1]
        all_query_scores = []
        for query_contexts_ranked, query in zip(all_query_docs_sorted, questions):
            features = rank_model_tokenizer(
                [query] * buffer_size,
                [normalization_txt(format(*context)) for context in query_contexts_ranked[:buffer_size]],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(0)
            all_query_scores.append(rank_model(**features).logits.transpose(0, 1)[0])
    else:
        all_query_scores = [
            list(
                models[1].predict(
                    [(query, normalization_txt(format(*context))) for context in query_contexts_ranked[:buffer_size]]
                )
            )
            for query_contexts_ranked, query in zip(all_query_docs_sorted, questions)
        ]

    all_query_docsandscores_sorted = [
        sorted(list(zip(query_contexts_ranked[:buffer_size], query_scores[:buffer_size])), key=lambda x: x[1], reverse=True)
        for query_contexts_ranked, query_scores in zip(all_query_docs_sorted, all_query_scores)
    ]
    all_query_docs_sorted = [
        [docandscore[0] for docandscore in query_docsandscores_sorted][:neighbors]
        for query_docsandscores_sorted in all_query_docsandscores_sorted
    ]
    similarity = [[item[1].item() for item in batch_doc_scores][:neighbors] for batch_doc_scores in all_query_docsandscores_sorted]
    return all_query_docs_sorted, similarity


class ContriverRetrievalResource(Resource):
    """
    BM25 Retrieval Flask resource.
    The PUT method is to get KNN tokens, add new chunks, reset index.
    """

    def __init__(self, retriever, all_context, all_context_emb, tokenizer, pad_len, models):
        self.retriever = retriever
        self.all_context = all_context
        self.all_context_emb = all_context_emb
        self.tokenizer = tokenizer
        self.pad_len = pad_len
        self.models = models

    def put(self):
        data = request.get_json()
        if 'neighbors' in data:
            sentences = data['sentences']
            # do knn query
            num_neighbors = data['neighbors']
            with lock:  # Need to get lock to keep multiple threads from hitting code
                all_neighbors = []
                results, similarity = get_relevant_context(
                    sentences, self.all_context, self.all_context_emb, num_neighbors, self.models
                )
                all_first_neighbors = []
                for relevant_context_and_title in results:
                    token_ids = []
                    # the first neighbor is not trunked
                    first_neighbor = []
                    count = 0
                    for context, title in relevant_context_and_title:
                        item = format(context, title)
                        ids = self.tokenizer.text_to_ids(item)
                        if count == 0:
                            first_neighbor.append(ids)
                        count += 1
                        if len(ids) < self.pad_len:
                            ids = ids + [self.tokenizer.eos_id] * (self.pad_len - len(ids))
                        elif len(ids) > self.pad_len:
                            ids = ids[: self.pad_len]
                        token_ids.append(ids)
                    all_neighbors.append(token_ids)
                    all_first_neighbors.append(first_neighbor)
            result = {'knn': all_neighbors, "similarity": similarity, 'first_neighbor': all_first_neighbors}
            return jsonify(result)
        return "wrong API"


class ContriverRetrievalServer(object):
    """
    Flask Retrieval server, which helps to get the KNN tokens given the query chunk
    """

    def __init__(self, filepath, tokenizer: TokenizerSpec, max_answer_length, cross_encoder=None):
        self.app = Flask(__name__, static_url_path='')
        rows = load_car_manual(filepath)
        manual = []
        for i, row in enumerate(rows):
            text = ftfy.fix_text(row['text'].strip())
            title = row['title'].strip()
            manual.append((text, title))
        all_contexts = chunk_car_manual(manual)
        all_contexts_formatted = [normalization_txt(format(*all_context)) for all_context in all_contexts]
        self.distil_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b', device='cuda')
        if cross_encoder is not None:
            self.rank_model = torch.load(cross_encoder).eval().cuda()
            rank_model_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
            reranker = (self.rank_model, rank_model_tokenizer)
        else:
            self.rank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512, device='cuda')
            reranker = self.rank_model
        self.all_contexts_formatted_emb = self.distil_model.encode(all_contexts_formatted)
        self.all_contexts = all_contexts

        all_retrievers = {
            "bm25": {"retriever_fcn": bm25_retriever},
            'msmarcominilm': {"retriever_fcn": msmarcominilm_retriever},
        }
        all_retrievers.update(
            {
                "tasb": {
                    "retriever_fcn": tasb_retriever_msmarcominilm_reranker,
                    "retriever_args": {"retrieval_method": "DOT_PRODUCT"},
                }
            }
        )
        retriever_name = "tasb"
        self.retriever = all_retrievers[retriever_name]

        api = Api(self.app)
        api.add_resource(
            ContriverRetrievalResource,
            '/knn',
            resource_class_args=[
                self.retriever,
                self.all_contexts,
                self.all_contexts_formatted_emb,
                tokenizer,
                max_answer_length,
                (self.distil_model, reranker),
            ],
        )

    def run(self, url, port=None):
        threading.Thread(target=lambda: self.app.run(host=url, threaded=True, port=port)).start()
