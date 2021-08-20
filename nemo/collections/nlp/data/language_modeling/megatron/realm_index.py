import itertools
import os
import pickle
import shutil

import numpy as np
import torch

from megatron import get_args
from megatron import mpu


def detach(tensor):
    return tensor.detach().cpu().numpy()


class OpenRetreivalDataStore(object):
    """
    Serializable data structure for holding data for blocks --
    embeddings and necessary metadata for Retriever
    """
    def __init__(self, embedding_path=None, load_from_path=True, rank=None):
        self.embed_data = dict()
        if embedding_path is None:
            args = get_args()
            embedding_path = args.embedding_path
            rank = args.rank
        self.embedding_path = embedding_path
        self.rank = rank

        if load_from_path:
            self.load_from_file()

        block_data_name = os.path.splitext(self.embedding_path)[0]
        self.temp_dir_name = block_data_name + '_tmp'

    def state(self):
        return {
            'embed_data': self.embed_data,
        }

    def clear(self):
        """
        Clear the embedding data structures to save memory.
        The metadata ends up getting used, and is also much smaller in
        dimensionality so it isn't really worth clearing.
        """
        self.embed_data = dict()

    def load_from_file(self):
        """Populate members from instance saved to file"""

        if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
            print("\n> Unpickling BlockData", flush=True)
        state_dict = pickle.load(open(self.embedding_path, 'rb'))
        if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
            print(">> Finished unpickling BlockData\n", flush=True)

        self.embed_data = state_dict['embed_data']

    def add_block_data(self, row_id, block_embeds, allow_overwrite=False):
        """
        Add data for set of blocks
        :param row_id: 1D array of unique int ids for the blocks
        :param block_embeds: 2D array of embeddings of the blocks
            In the case of retriever this will be [start_idx, end_idx, doc_idx]
        """
        for idx, embed in zip(row_id, block_embeds):
            if not allow_overwrite and idx in self.embed_data:
                raise ValueError("Unexpectedly tried to overwrite block data")

            self.embed_data[idx] = np.float16(embed)

    def save_shard(self):
        """
        Save the block data that was created this in this process
        """
        if not os.path.isdir(self.temp_dir_name):
            os.makedirs(self.temp_dir_name, exist_ok=True)

        # save the data for each shard
        with open('{}/{}.pkl'.format(self.temp_dir_name, self.rank), 'wb') \
            as writer:
            pickle.dump(self.state(), writer)

    def merge_shards_and_save(self):
        #Combine all the shards made using save_shard
        shard_names = os.listdir(self.temp_dir_name)
        seen_own_shard = False

        for fname in os.listdir(self.temp_dir_name):
            shard_rank = int(os.path.splitext(fname)[0])
            if shard_rank == self.rank:
                seen_own_shard = True
                continue

            with open('{}/{}'.format(self.temp_dir_name, fname), 'rb') as f:
                data = pickle.load(f)
                old_size = len(self.embed_data)
                shard_size = len(data['embed_data'])

                # add the shard's data and check to make sure there
                # is no overlap
                self.embed_data.update(data['embed_data'])
                assert len(self.embed_data) == old_size + shard_size

        assert seen_own_shard

        # save the consolidated shards and remove temporary directory
        with open(self.embedding_path, 'wb') as final_file:
            pickle.dump(self.state(), final_file)
        shutil.rmtree(self.temp_dir_name, ignore_errors=True)

        print("Finished merging {} shards for a total of {} embeds".format(
            len(shard_names), len(self.embed_data)), flush=True)


class FaissMIPSIndex(object):
    """
    Wrapper object for a BlockData which similarity search via FAISS under the hood
    """
    def __init__(self, embed_size, embed_data=None, use_gpu=False):
        self.embed_size = embed_size
        self.embed_data = embed_data
        self.use_gpu = use_gpu

        self.mips_index = None
        self._set_mips_index()

    def _set_mips_index(self):
        """
        Create a Faiss Flat index with inner product as the metric
        to search against
        """
        try:
            import faiss
        except ImportError:
            raise Exception("Error: Please install faiss to use FaissMIPSIndex")

        if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
            print("\n> Building index", flush=True)

        cpu_index = faiss.IndexFlatIP(self.embed_size)

        if self.use_gpu:
            # create resources and config for GpuIndex
            config = faiss.GpuMultipleClonerOptions()
            config.shard = True
            config.useFloat16 = True
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=config)
            self.mips_index = faiss.IndexIDMap(gpu_index)
            if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
                print(">> Initialized index on GPU", flush=True)
        else:
            # CPU index supports IDs so wrap with IDMap
            self.mips_index = faiss.IndexIDMap(cpu_index)
            if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
                print(">> Initialized index on CPU", flush=True)

        # if we were constructed with a BlockData, then automatically load it
        # when the FAISS structure is built
        if self.embed_data is not None:
            self.add_embed_data(self.embed_data)

    def reset_index(self):
        """Delete existing index and create a new"""
        del self.mips_index

        # reset the block data so that _set_block_index will reload it as well
        if self.embed_data is not None:
            embed_data_path = self.embed_data.embedding_path
            del self.embed_data
            self.embed_data = OpenRetreivalDataStore(embed_data_path)

        self._set_mips_index()

    def update_index(self):
        """Delete existing index and create a new"""
        del self.mips_index

        # reset the block data so that _set_mips_index will reload it as well
        if self.embed_data is not None:
            self.embed_data.load_from_file()
        self._set_mips_index()

    def add_embed_data(self, all_embed_data):
        """Add the embedding of each block to the underlying FAISS index"""

        # this assumes the embed_data is a dict : {int: np.array<float>}
        block_indices, block_embeds = zip(*all_embed_data.embed_data.items())

        # the embeddings have to be entered in as float32 even though the math
        # internally is done with float16.
        embeds_arr = np.float32(np.array(block_embeds))
        indices_arr = np.array(block_indices)

        # we no longer need the embedding data since it's in the index now
        all_embed_data.clear()

        self.mips_index.add_with_ids(embeds_arr, indices_arr)

        if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
            print(">>> Finished adding block data to index", flush=True)

    def search_mips_index(self, query_embeds, top_k, reconstruct=True):
        """
        Get the top-k blocks by the index distance metric.

        :param reconstruct: if True: return a [num_queries x k x embed_dim]
                                array of blocks
                            if False: return [num_queries x k] array of
                                distances, and another for indices
        """
        query_embeds = np.float32(detach(query_embeds))

        if reconstruct:
            # get the vectors themselves
            top_k_block_embeds = self.mips_index.search_and_reconstruct(\
                query_embeds, top_k)
            return top_k_block_embeds
        else:
            # get distances and indices of closest vectors
            distances, block_indices = self.mips_index.search(query_embeds, top_k)
            return distances, block_indices
