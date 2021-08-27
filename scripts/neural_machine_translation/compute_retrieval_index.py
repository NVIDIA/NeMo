from logging import error
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import os.path

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=int, default=1, help="1: bert base indices, 2: paraphrase")
    parser.add_argument("--nns_to_save", type=int, default=10, help="Nearest neighbors to save in index")
    parser.add_argument("--query", type=str, required=True, help='Query file for which nearest neighbors are computed.')
    parser.add_argument("--index_src_main", type=str, help='File with index src. Corresponds to embeddings_main')
    parser.add_argument("--index_src_additional", type=str, help='File with additional index src. Corresponds to embeddings_main_additional.')
    parser.add_argument("--embeddings_main", type=str, help='Embeddings for the train file')
    parser.add_argument("--embeddings_additional", type=str, help='Embeddings for the additional index')
    parser.add_argument("--save_path", type=str, required=True, help='Path to save the indices file as .npy')
    return parser.parse_args()

def compute_embeddings(model, file, save_path):
    line_ctr = 0
    # Create empty embedding np array
    with open(args.file) as f:
        for line in f:
            line_ctr += 1
    embeddings = np.zeros((line_ctr, 768)).astype(np.float32)

    counter = 0
    cur_buffer = []
    idx_ctr = 0
    for idx, line in enumerate(open(file)):
        cur_buffer.append(line)
        counter += 1
        if counter == 1000:
            with torch.no_grad():
                cur_embeddings = model.encode(cur_buffer, show_progress_bar=False)
                cur_buffer = []
                counter = 0
                for emb in cur_embeddings:
                    embeddings[idx_ctr] = emb
                    idx_ctr += 1
        if idx % 10000 == 0:
            print('Finished %d lines' % (idx))

    with torch.no_grad():
        cur_embeddings = model.encode(cur_buffer, show_progress_bar=False)
        cur_buffer = []
        counter = 0
        for emb in cur_embeddings:
            embeddings[idx_ctr] = emb
            idx_ctr += 1
    np.save(save_path, embeddings)
    return embeddings

def faiss_search_vanilla(gpu_index_flat, query, k):
    D, I = gpu_index_flat.search(query, k)
    return I

if __name__ == '__main__':
    args = get_args()
    if args.model == 1:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
    elif args.model == 2:
        model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')
    else:
        raise NotImplementedError
    model = model.cuda()

    if os.path.isfile(args.embeddings_main):
        print('Embeddings already exist')
        embeddings = np.load(args.embeddings_main)
    else:
        print('Computing embeddings')
        embeddings = compute_embeddings(model, args.index_src_main, args.embeddings_main)
    
    if args.index_src_additional is not None:
        # Expand the index
        if os.path.isfile(args.embeddings_additional):
            print('Additional Embeddings already exist')
            embeddings_add = np.load(args.embeddings_additional)
        else:
            print('Computing additional embeddings')
            embeddings_add = compute_embeddings(model, args.index_src_additional, args.embeddings_additional)
        embeddings = np.concatenate((embeddings, embeddings_add))

    print("number of GPUs:", faiss.get_num_gpus())
    index = faiss.IndexFlatIP(embeddings.shape[1])

    # res = faiss.StandardGpuResources()  # use a single GPU
    # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    # gpu_index_flat = faiss.index_cpu_to_all_gpus(index)

    gpu_index_flat = faiss.index_cpu_to_gpus_list(index, gpus=[0])
    # gpu_index_flat = faiss.index_cpu_to_gpus_list(index, gpus=[1,2,3])
    gpu_index_flat.add(embeddings)
    print(gpu_index_flat.ntotal)

    lines = [line.strip() for line in open(args.query)]

    nn_list = []
    for i in tqdm(range(0, len(lines), 2000)):
        reference = lines[i:i+2000]
        query = model.encode(reference, show_progress_bar=False)
        selected_idxs = faiss_search_vanilla(gpu_index_flat, query, args.nns_to_save)
        nn_list.append(selected_idxs)
    
    indexes = np.concatenate(nn_list, axis=0)
    print(indexes.shape)

    with open(args.save_path, 'wb') as f:
        np.save(f, indexes)