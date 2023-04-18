import argparse
import json
import threading

import ftfy
import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.modules.common.megatron.retrieval_services.contriever_server import format
from nemo.collections.nlp.modules.common.megatron.retrieval_services.util import lock, request_data


def normalization_txt(text):
    return text.lower()


def get_similar_idx_val(embed_text, dpr_group, topk_count, text_retriever_group, use_bm_25=False, bm_model=None):
    (dpr_model, dpr_tokenizer) = dpr_group
    (retriever_embeddings, _) = text_retriever_group
    embed_ids = dpr_tokenizer.encode(embed_text)
    embed_ids = torch.LongTensor([embed_ids]).cuda()
    embed_emb = dpr_model(input_ids=embed_ids).pooler_output
    embed_emb = embed_emb[0]
    similarity_list = retriever_embeddings.matmul(embed_emb)

    if use_bm_25:
        tokenized_query = normalization_txt(embed_text).split()
        doc_scores = bm_model.get_scores(tokenized_query)
        all_valid_idx_1 = (-doc_scores).argsort()[:2]
        all_valid_idx_2 = (-doc_scores).argsort()[2:10]
        values, indices = torch.sort(similarity_list, descending=True)
        valid_v_i_1 = [(v, i) for v, i in zip(values, indices) if i.item() in all_valid_idx_1]
        valid_v_i_2 = [(v, i) for v, i in zip(values, indices) if i.item() in all_valid_idx_2]
        valid_v_i = valid_v_i_1 + valid_v_i_2
        assert len(valid_v_i) > 0
        values = [v for v, i in valid_v_i[:topk_count]]
        indices = [i for v, i in valid_v_i[:topk_count]]
    else:
        values, indices = torch.topk(similarity_list, topk_count, largest=True)
    return values, indices


def get_relevant_context(embed_text, dpr_group, context_retriever_group, topk_count=1, two_retriever=True):

    if two_retriever:
        from rank_bm25 import BM25Okapi

        _, context_text = context_retriever_group
        tokenized_corpus = [normalization_txt(t + c).split() for t, c in context_text]
        bm25 = BM25Okapi(tokenized_corpus)
        values, indices = get_similar_idx_val(embed_text, dpr_group, topk_count, context_retriever_group, True, bm25)
    else:
        values, indices = get_similar_idx_val(embed_text, dpr_group, topk_count, context_retriever_group)

    (_, retriever_dataset_ext) = context_retriever_group
    # relevant_context_and_title = retriever_dataset_ext[indices[0]]
    relevant_context_and_title = [retriever_dataset_ext[ind] for ind in indices]

    return relevant_context_and_title


def add_retrieve_args(parser):
    """Car manual QA arguments."""
    group = parser.add_argument_group(title='car manual qa')
    group.add_argument('--car-manual', type=str, help='Path to the car manual')
    group.add_argument('--question-answer', type=str, help='Path to the question-answer dataset')
    group.add_argument('--predict-ground-path', type=str, help='Path to write the predictions')
    group.add_argument("--sliding-window-length", type=int, default=368, help='Sliding window length for context')
    group.add_argument("--sliding-length", type=int, default=64, help='Sliding length for context')
    group.add_argument("--few-shot-qa", type=int, default=24, help='Few shot samples QA to use from the retriever')
    group.add_argument(
        '--question-answer-splits',
        type=str,
        default='80, 20',
        help='Comma-separated list of proportions for training,' ' and test split.',
    )
    group.add_argument("--max-answer-length", type=int, default=128, help='max length of the generated answer.')
    group.add_argument("--top_k", type=int, default=0, help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument("--beam_size", type=int, default=0, help='beam size.')
    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument(
        '--retriever-qa-type',
        type=str,
        default='random',
        choices=['similar', 'random'],
        help='Use highest or random retrieved documents',
    )
    group.add_argument(
        '--megatron-api-url', type=str, default='http://10.14.74.235:5000/api', help='url of the megatron api'
    )
    group.add_argument('--use-megatron-api', action='store_true', default=None, help='Use megatron api of 530B model')
    group.add_argument(
        '--retriever-context-type',
        type=str,
        default='groundtruth',
        choices=['groundtruth', 'knowledgebase', 'blendcontext'],
        help='What is the sources of context to use',
    )
    group.add_argument('--nctx', type=int, default=1, help='number of ctx documents')
    group.add_argument('--use-context-title', action='store_true', default=None, help='Use context title')
    group.add_argument('--header-context', type=str, default='', help='Additional contexts to use for the headings')
    group.add_argument('--mode', type=str, default='single', help='dpr model mode')
    group.add_argument(
        '--prefix-question-with-prompt', action='store_true', default=None, help='prefix question with prompt or not'
    )
    group.add_argument('--prefix-idx', type=int, default=0, help='prefix index to choose prompt')
    group.add_argument(
        '--prefix-with-prompt', action='store_true', default=None, help='prefix question with prompt or not'
    )
    group.add_argument('--prefix-ending', action='store_true', default=None, help='prefix at the end or not')
    group.add_argument(
        '--add-title-to-retriever', action='store_true', default=None, help='if we should add title to retriever'
    )
    group.add_argument('--two-retriever', action='store_true', default=None, help='if we should use two retriever')
    group.add_argument('--cqa', action='store_true', default=None, help='if we should the format of cqa or not')
    group.add_argument('--output-dir', type=str, default=None, help='output dir to write')
    return parser


def get_retriever_dataset(car_manual):
    data = []
    context_titles = []

    with open(car_manual, 'r') as f:
        rows = json.load(f)
        for row in rows:
            text = ftfy.fix_text(row['text'].strip())
            title = row['title'].strip()
            example = text
            data.append(example)
            context_titles.append((title))

    return data, context_titles


def add_embedding(cur_ids, dpr_model):
    embed_ids = torch.LongTensor([cur_ids]).cuda()
    embed_emb = dpr_model(input_ids=embed_ids).pooler_output
    return embed_emb


def get_text_from_id(embed_ids, dpr_tokenizer):
    decode_text = dpr_tokenizer.decode(embed_ids)
    # decode_text = decode_text[len('[CLS] '): \
    #    len(decode_text)-len(' [SEP]')]
    return decode_text


def get_dpr_group(mode="single"):
    if mode == "single":
        dpr_path = "facebook/dpr-ctx_encoder-single-nq-base"
    elif mode == "multi":
        dpr_path = "facebook/dpr-ctx_encoder-multiset-base"
    else:
        raise ValueError("wrong mode for dpr")
    print(dpr_path)
    print('Building embeddings using DPR context encoder')
    dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_path)
    dpr_model = DPRContextEncoder.from_pretrained(dpr_path).cuda()

    return (dpr_model, dpr_tokenizer)


def get_bert_group():

    from transformers import BertModel, BertTokenizer

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").cuda()
    return bert_model, bert_tokenizer


def get_retriever_embeddings(args, retriever_dataset, dpr_group, context_titles=None):

    (dpr_model, dpr_tokenizer) = dpr_group

    retriever_dataset_ext = []

    with torch.no_grad():
        for idx, data_instance in enumerate(retriever_dataset):
            embed_text = data_instance
            embed_ids = dpr_tokenizer.encode(embed_text)
            title = ""
            if context_titles is not None:
                title = context_titles[idx]
                title_ids = dpr_tokenizer.encode(title)

            # this is sliding window when the seqence is large
            if len(embed_ids) > args.sliding_window_length:
                start_idx, end_idx = 0, args.sliding_window_length
                isEnd = False
                while True:
                    if args.add_title_to_retriever and context_titles is not None:
                        cur_ids = title_ids + embed_ids[start_idx:end_idx]
                    else:
                        cur_ids = embed_ids[start_idx:end_idx]
                    embed_emb = add_embedding(cur_ids, dpr_model)
                    retriever_embeddings = (
                        torch.cat((retriever_embeddings, embed_emb), dim=0) if idx > 0 else embed_emb
                    )

                    example = get_text_from_id(cur_ids, dpr_tokenizer)
                    retriever_dataset_ext.append((example, title))

                    if isEnd:
                        break

                    start_idx += args.sliding_length
                    end_idx += args.sliding_length
                    if end_idx > len(embed_ids):
                        end_idx = len(embed_ids)
                        isEnd = True
            else:
                if args.add_title_to_retriever and context_titles is not None:
                    embed_ids = title_ids + embed_ids
                embed_emb = add_embedding(embed_ids, dpr_model)
                retriever_embeddings = torch.cat((retriever_embeddings, embed_emb), dim=0) if idx > 0 else embed_emb

                example = get_text_from_id(embed_ids, dpr_tokenizer)
                retriever_dataset_ext.append((example, title))

    return (retriever_embeddings, retriever_dataset_ext)


def load_predefined_args():

    parser = argparse.ArgumentParser()
    parser = add_retrieve_args(parser)
    args = parser.parse_args([])
    # args. =

    return args


def test(query):

    args = load_predefined_args()

    dpr_group = get_dpr_group(args.mode)
    bert_group = get_bert_group()

    context_retriever_dataset, context_titles = get_retriever_dataset(args.car_manual)
    context_retriever_group = get_retriever_embeddings(args, context_retriever_dataset, dpr_group, context_titles)

    question = query
    relevant_context_and_title = get_relevant_context(
        question, bert_group, context_retriever_group, topk_count=10, two_retriever=args.two_retriever
    )
    print(relevant_context_and_title)
    return relevant_context_and_title


class BM25RetrievalResource(Resource):
    """
    BM25 Retrieval Flask resource.
    The PUT method is to get KNN tokens, add new chunks, reset index.
    """

    def __init__(self, bert_group, context_retriever_group, tokenizer, pad_len):
        self.bert_group = bert_group
        self.context_retriever_group = context_retriever_group
        self.tokenizer = tokenizer
        self.pad_len = pad_len

    def put(self):
        data = request.get_json()
        if 'neighbors' in data:
            sentences = data['sentences']
            # do knn query
            num_neighbors = data['neighbors']
            with lock:  # Need to get lock to keep multiple threads from hitting code
                all_neighbors = []
                for sentence in sentences:
                    question = sentence
                    # the first neighbor is not trunked
                    first_neighbor = []
                    count = 0
                    relevant_context_and_title = get_relevant_context(
                        question,
                        self.bert_group,
                        self.context_retriever_group,
                        topk_count=num_neighbors,
                        two_retriever=True,
                    )
                    token_ids = []
                    for context, title in relevant_context_and_title:
                        item = format(context, title)
                        ids = self.tokenizer.text_to_ids(item)
                        if count == 0:
                            first_neighbor.append(ids)
                        count += 1
                        if len(ids) < self.pad_len:
                            ids = ids + [self.tokenizer.eos_id] * len(ids) * (self.pad_len - len(ids))
                        elif len(ids) > self.pad_len:
                            ids = ids[: self.pad_len]
                        token_ids.append(ids)
                    all_neighbors.append(token_ids)
            result = {'knn': all_neighbors, 'first_neighbor': first_neighbor}
            return jsonify(result)
        return "wrong API"


class BM25RetrievalServer(object):
    """
    Flask Retrieval server, which helps to get the KNN tokens given the query chunk
    """

    def __init__(
        self, filepath, sliding_window_length, sliding_length, max_answer_length, nctx, tokenizer: TokenizerSpec,
    ):
        self.app = Flask(__name__, static_url_path='')
        args = load_predefined_args()
        args.car_manual = filepath
        args.sliding_window_length = sliding_window_length
        args.sliding_length = sliding_length
        args.max_answer_length = max_answer_length
        args.nctx = nctx
        args.two_retriever = True
        args.use_context_title = True

        self.dpr_group = get_dpr_group(args.mode)
        self.bert_group = get_bert_group()

        self.context_retriever_dataset, context_titles = get_retriever_dataset(args.car_manual)
        self.context_retriever_group = get_retriever_embeddings(
            args, self.context_retriever_dataset, self.dpr_group, context_titles
        )
        api = Api(self.app)
        api.add_resource(
            BM25RetrievalResource,
            '/knn',
            resource_class_args=[self.bert_group, self.context_retriever_group, tokenizer, max_answer_length],
        )

    def run(self, url, port=None):
        threading.Thread(target=lambda: self.app.run(host=url, threaded=True, port=port)).start()