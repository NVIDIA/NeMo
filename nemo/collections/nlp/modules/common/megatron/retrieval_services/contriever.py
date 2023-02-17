import torch
from pynvml import *
from transformers import AutoModel, AutoTokenizer


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def compute_embedding(tokenizer, model, sentences):
    # Apply tokenizer
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    inputs = inputs.to("cuda")
    # Compute token embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    return mean_pooling(outputs[0], inputs['attention_mask'])


def contriever_group():
    # Get contriever model and tokenizer
    # model_name = 'contriever'
    model_name = 'contriever-msmarco'
    # model_name = 'mcontriever'
    # model_name = 'mcontriever-msmarco'

    path = 'facebook/' + model_name
    contriever_tokenizer = AutoTokenizer.from_pretrained(path)
    contriever_model = AutoModel.from_pretrained(path).to("cuda")
    contriever_model.eval()
    print_gpu_utilization()
    return contriever_model, contriever_tokenizer


def contriever_retriever(all_contexts, queries):
    # For every query get a sorted list of all_contexts based on similarity
    print('Contriever retriever')
    contriever_model, contriever_tokenizer = contriever_group()
    query_embeddings = compute_embedding(contriever_tokenizer, contriever_model, queries)
    print_gpu_utilization()
    bsz = 256
    to_encode = []
    all_contexts_embeddings = []
    for i in range(len(all_contexts)):
        to_encode.append(all_contexts[i][1] + ': ' + all_contexts[i][0])
        if len(to_encode) == bsz:
            all_contexts_embeddings.append(compute_embedding(contriever_tokenizer, contriever_model, to_encode))
            print_gpu_utilization()
            to_encode = []
    if len(to_encode) > 0:
        all_contexts_embeddings.append(compute_embedding(contriever_tokenizer, contriever_model, to_encode))

    all_contexts_embeddings = torch.cat(all_contexts_embeddings, dim=0)
    scores = torch.matmul(query_embeddings, all_contexts_embeddings.T)
    ranked_docs = []
    for i in range(len(queries)):
        doc_rank = (-scores[i]).argsort()
        ranked_docs.append([all_contexts[j] for j in doc_rank])
    return ranked_docs
