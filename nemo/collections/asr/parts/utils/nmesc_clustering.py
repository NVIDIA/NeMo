# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2007-2020 The scikit-learn developers.

# BSD 3-Clause License

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file is part of https://github.com/scikit-learn/scikit-learn/blob/114616d9f6ce9eba7c1aacd3d4a254f868010e25/sklearn/manifold/_spectral_embedding.py and
# https://github.com/tango4j/Auto-Tuning-Spectral-Clustering.

from collections import Counter

import numpy as np
from tqdm import tqdm

from sklearn.cluster._kmeans import k_means, kmeans_plusplus
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.metrics.pairwise import cosine_similarity, _euclidean_distances
from sklearn.preprocessing import MinMaxScaler

from nemo.utils import logging
from nemo.utils.decorators.experimental import experimental
from functools import partial
import torch
from typing import Dict, List
scaler = MinMaxScaler(feature_range=(0, 1))
DEVICE = 0
try:
    from torch.linalg import eigh as eigh

    TORCH_EIGN = True

except ImportError:
    TORCH_EIGN = False
    from scipy.linalg import eigh as eigh

    logging.warning("Using eigen decomposition from scipy, upgrade torch to 1.9 or higher for faster clustering")


@torch.jit.script
def cos_similarity(a, b):
    eps = torch.tensor(0.00035)
    a_norm = a / (torch.norm(a, dim=1).unsqueeze(1) + eps)
    b_norm = b / (torch.norm(a, dim=1).unsqueeze(1) + eps)
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    res.fill_diagonal_(1)
    return res

@torch.jit.script
def ScalerMinMax(X):
    v_min, v_max = X.min(dim=0)[0], X.max(dim=0)[0]
    v_min, v_max = v_min.expand(X.shape[0], -1), v_max.expand(X.shape[0], -1)
    v_std = (X - v_min)/(v_max - v_min)
    v_p = v_std + 1.0
    return v_p

@torch.jit.script
def getEuclideanDistance(specEmbA, specEmbB, device:torch.device=torch.device('cpu')): 
    specEmbA, specEmbB = specEmbA.to(device), specEmbB.to(device)
    A, B= specEmbA.unsqueeze(dim=1), specEmbB.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis

@torch.jit.script
def kmeans_plusplus_torch(X, n_clusters: int, random_state: int, n_local_trials: int=10, device: torch.device=torch.device('cpu')):
    torch.manual_seed(random_state)
    X = X.to(device)
    x_squared_norms = torch.einsum("ij,ij->i", X, X)
    n_samples, n_features = X.shape
    
    centers = torch.zeros(n_clusters, n_features, dtype=X.dtype)
    # Set the number of local seeding trials if none is given
    # Pick first center randomly and track index of point
    # center_id = torch.randint(0, n_samples, (1, )).item()
    center_id = torch.randint(0, n_samples, (1, )).long()
    indices = torch.full([n_clusters, ], -1, dtype=torch.int)

    centers[0] = X[center_id].squeeze(0)
    indices[0] = center_id.squeeze(0)

    centers = centers.to(device)
    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = (centers[0, None].repeat(1, X.shape[0]).view(X.shape[0],-1) - X).pow(2).sum(dim=1).unsqueeze(dim=0)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = torch.rand(n_local_trials) * current_pot.item()

        if len(closest_dist_sq.shape) > 1:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=1)[0]
        else:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=0)
            
        candidate_ids = torch.searchsorted(torch_cumsum, rand_vals.to(device))

        # XXX: numerical imprecision can result in a candidate_id out of range
        # Compute distances to center candidates
        N_ci = candidate_ids.shape[0]
        distance = (X[candidate_ids].repeat(1, X.shape[0]).view(X.shape[0]*N_ci,-1) - X.repeat(N_ci,1)).pow(2).sum(dim=1).view(N_ci, -1)


        # update closest distances squared and potential for each candidate
        distance_to_candidates = torch.minimum(closest_dist_sq, distance)
        candidates_pot = distance_to_candidates.sum(dim=1)

        # Decide which candidate is the best
        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]
        
        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices

@torch.jit.script
def kmeans_torch(
        X,
        num_clusters: int,
        tol: float=1e-4,
        iter_limit: int=15,
        device: torch.device=torch.device('cpu'),
):
    print(f'running k-means on {device}..')   
    # convert to float
    X = X.float().to(device)

    init_state = kmeans_plusplus_torch(X, n_clusters=num_clusters, random_state=0, device=device)
    initial_state = init_state[0]
    
    iter_count = 0
    selected_cluster_index = torch.tensor(0)
    while True:
        euc_dist = getEuclideanDistance(X, initial_state, device=device)
        selected_cluster_index = torch.argmin(euc_dist, dim=1)
        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(selected_cluster_index == index).squeeze().to(device)
            selected = torch.index_select(X, 0, selected)

            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iter_count
        iter_count += 1
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iter_count >= iter_limit:
            break

    return selected_cluster_index.cpu(), initial_state.cpu()


@torch.jit.script
def getTheLargestComponent(affinity_mat, seg_index: int, device: torch.device):
    """
    Find the largest affinity_mat connected components for each given node.
    This is for checking whether the affinity_mat is fully connected.
    """
    num_of_segments = affinity_mat.shape[0]

    connected_nodes = torch.zeros(num_of_segments, dtype=torch.bool).to(device)
    nodes_to_explore = torch.zeros(num_of_segments, dtype=torch.bool).to(device)

    nodes_to_explore[seg_index] = True
    for k in range(num_of_segments):
        last_num_component = connected_nodes.sum()
        torch.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break

        indices = (nodes_to_explore == torch.tensor(True)).nonzero().t().squeeze()
        if len(indices.size()) == 0:
            indices = indices.unsqueeze(0)
        for i in indices:
            neighbors = affinity_mat[i]
            torch.logical_or(nodes_to_explore, neighbors.squeeze(0), out=nodes_to_explore)
    return connected_nodes

@torch.jit.script
def isGraphFullyConnected(affinity_mat, device: torch.device):
    return getTheLargestComponent(affinity_mat, 0, device).sum() == affinity_mat.shape[0]



@torch.jit.script
def getKneighborsConnections(affinity_mat, p_value: int):
    """
    Binarize top-p values for each row from the given affinity matrix.
    """
    binarized_affinity_mat = torch.zeros_like(affinity_mat)
    for i in range(affinity_mat.shape[0]):
        line = affinity_mat[i, :]
        sorted_idx = torch.argsort(line, descending=True)
        indices = sorted_idx[:p_value]
        binarized_affinity_mat[indices, i] = 1

    return binarized_affinity_mat

@torch.jit.script
def getAffinityGraphMat(affinity_mat_raw, p_value: int):
    """
    Calculate a binarized graph matrix and
    symmetrize the binarized graph matrix.
    """
    X = getKneighborsConnections(affinity_mat_raw, p_value)
    symm_affinity_mat = 0.5 * (X + X.T)
    return symm_affinity_mat


@torch.jit.script
def getMinimumConnection(mat, max_N, n_list, device: torch.device):
    """
    Generate connections until fully connect all the nodes in the graph.
    If graph is not fully connected, it might generate an inaccurate results.
    """
    p_value  = torch.tensor(1)
    affinity_mat = getAffinityGraphMat(mat, p_value)
    for i, p_value in enumerate(n_list):
        fully_connected = isGraphFullyConnected(affinity_mat, device)
        affinity_mat = getAffinityGraphMat(mat, p_value)
        if fully_connected or p_value > max_N:
            break

    return affinity_mat, p_value

@torch.jit.script
def _getRepeatedList(mapping_argmat, score_mat_size):
    """
    Count the numbers in the mapping dictionary and create lists that contain
    repeated indices to be used for creating the repeated affinity matrix for
    fusing the affinity values.
    """
    repeat_list = torch.zeros(score_mat_size, dtype=torch.int32)
    idxs, counts = torch.unique(mapping_argmat, return_counts=True)
    repeat_list[idxs] = counts.int()
    return repeat_list

@torch.jit.script
def get_argmin_mat(uniq_scale_list: List[List[torch.Tensor]]):
    """
    Calculate the mapping between the base scale and other scales. A segment from a longer scale is
    repeatedly mapped to a segment from a shorter scale or the base scale.

    Args:
        uniq_scale_dict (dict) :
            Dictionary of embeddings and timestamps for each scale.

    Returns:
        session_scale_mapping_dict (dict) :
            Dictionary containing argmin arrays indexed by scale index.
    """
    N_scale = len(uniq_scale_list)
    segment_anchor_dict = [torch.tensor(0)] * N_scale
    for scale_idx in range(N_scale):
        time_stamps_float = uniq_scale_list[scale_idx][1]
        segment_anchor_dict[scale_idx] = torch.mean(time_stamps_float, dim=1)

    base_scale_idx = N_scale - 1
    base_scale_anchor = segment_anchor_dict[base_scale_idx]
    session_scale_mapping_dict = [torch.tensor(0)] * N_scale
    for scale_idx in range(N_scale):
        curr_scale_anchor = segment_anchor_dict[scale_idx]
        curr_mat = torch.tile(curr_scale_anchor, (base_scale_anchor.shape[0], 1))
        base_mat = torch.tile(base_scale_anchor, (curr_scale_anchor.shape[0], 1)).t()
        argmin_mat = torch.argmin(torch.abs(curr_mat - base_mat), dim=1)
        session_scale_mapping_dict[scale_idx] = argmin_mat
    return session_scale_mapping_dict

@torch.jit.script
def _getMultiScaleCosAffinityMatrix(multiscale_weights: torch.Tensor, uniq_scale_list: List[List[torch.Tensor]], device: torch.device=torch.device('cpu')):
    """
    Calculate cosine similarity values among speaker embeddings for each scale then
    apply multiscale weights to calculate the fused similarity matrix.

    Args:
        uniq_embs_and_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_embs_and_timestamps contains only one scale, single scale diarization 
            is performed.

    Returns:
        fused_sim_d (np.array):
            This function generates an ffinity matrix that is obtained by calculating
            the weighted sum of the affinity matrices from the different scales.
        base_scale_emb (np.array):
            The base scale embedding (the embeddings from the finest scale)
    """
    N_scale = len(uniq_scale_list)
    base_scale_idx = N_scale - 1
    base_scale_emb = uniq_scale_list[base_scale_idx][0]
    multiscale_weights = multiscale_weights.unsqueeze(0).to(device)

    score_mat_list, repeated_tensor_list = [], []
    repeated_mat_list = [] 
    session_scale_mapping_dict = get_argmin_mat(uniq_scale_list)
    for scale_idx in range(N_scale):
        mapping_argmat = session_scale_mapping_dict[scale_idx]
        emb_t = uniq_scale_list[scale_idx][0].to(device)
        _emb = emb_t.half()
        sim_d = cos_similarity(_emb, _emb)
        score_mat_torch = ScalerMinMax(sim_d).to(device)
        repeat_list = _getRepeatedList(mapping_argmat, torch.tensor(score_mat_torch.shape[0])).to(device)
        repeated_tensor_0 = torch.repeat_interleave(score_mat_torch, repeats=repeat_list, dim=0)
        repeated_tensor_1 = torch.repeat_interleave(repeated_tensor_0, repeats=repeat_list, dim=1)
        repeated_tensor_list.append(repeated_tensor_1)
    repp = torch.stack(repeated_tensor_list).float().to(device)
    multiscale_weights = multiscale_weights.squeeze(0).float()
    fused_sim_d = torch.matmul(repp.permute(2,1,0), multiscale_weights.t()).squeeze(2).t()
    return fused_sim_d, base_scale_emb

@torch.jit.script
def getCosAffinityMatrix(_emb):
    """
    Calculate cosine similarity values among speaker embeddings.
    """
    _emb = _emb.half()
    sim_d = cos_similarity(_emb, _emb)
    sim_d = ScalerMinMax(sim_d)
    return sim_d

@torch.jit.script
def getLaplacian(X: torch.Tensor):
    """
    Calculate a laplacian matrix from an affinity matrix X.
    """
    X.fill_diagonal_(0)
    D = torch.sum(torch.abs(X), dim=1)
    D = torch.diag_embed(D)
    L = D - X
    return L

@torch.jit.script
def eigDecompose(laplacian, is_cuda: bool, device: torch.device=torch.device('cpu')):
    if is_cuda:
        if device is None:
            device = torch.cuda.current_device()
        laplacian = laplacian.float().to(device)
    else:
        laplacian = laplacian.float()
    lambdas, diffusion_map = eigh(laplacian)
    return lambdas, diffusion_map

@torch.jit.script
def getLamdaGaplist(lambdas):
    if torch.is_complex(lambdas):
        lambdas = torch.real(lambdas)
    return lambdas[1:] - lambdas[:-1]


@torch.jit.script
def estimateNumofSpeakers(affinity_mat, max_num_speaker: int, is_cuda: bool=False, device: torch.device=torch.device('cpu')):
    """
    Estimate the number of speakers using eigen decompose on laplacian Matrix.
    affinity_mat: (array)
        NxN affitnity matrix
    max_num_speaker: (int)
        Maximum number of clusters to consider for each session
    is_cuda: (bool)
        if cuda availble eigh decomposition would be computed on GPUs
    """
    if not is_cuda:
        device = torch.device('cpu')

    laplacian = getLaplacian(affinity_mat)
    lambdas, _ = eigDecompose(laplacian, is_cuda)
    lambdas = torch.sort(lambdas)[0]

    lambda_gap_tensor = getLamdaGaplist(lambdas)
    num_of_spk = torch.argmax(lambda_gap_tensor[: min(max_num_speaker, lambda_gap_tensor.shape[0])]) + 1
    return num_of_spk, lambdas, lambda_gap_tensor

@torch.jit.script
class SpectralClustering:
    def __init__(self, n_clusters:int=8, random_state:int=0, n_init:int=10, p_value:int=10, cuda:bool=False, device: torch.device=torch.device('cpu')):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.p_value = p_value
        self.cuda = cuda
        self.device = device

    def predict(self, X):
        if X.shape[0] != X.shape[1]:
            raise ValueError("The affinity matrix is not a square matrix.")
        labels = self.clusterSpectralEmbeddings(X, n_init=self.n_init, cuda=self.cuda, device=self.device)
        return labels

    def clusterSpectralEmbeddings(self, affinity, n_init: int=10, cuda: bool=False, device: torch.device=torch.device('cpu')):
        spectral_emb = self.getSpectralEmbeddings(affinity, n_spks=self.n_clusters, drop_first=False, cuda=cuda)
        labels, _ = kmeans_torch(
                    X=spectral_emb, num_clusters=self.n_clusters, device=device
                    )
        return labels
    
    def getSpectralEmbeddings(self, affinity_mat, n_spks:int=8, drop_first:bool=True, cuda:bool=False):
        laplacian = getLaplacian(affinity_mat)
        lambdas_, diffusion_map_ = eigDecompose(laplacian, cuda)
        diffusion_map = diffusion_map_[:, :n_spks]
        inv_idx = torch.arange(diffusion_map.size(1)-1, -1, -1).long()
        embedding = diffusion_map.T[inv_idx, :]
        return embedding[:n_spks].T

@torch.jit.script
class NMESC:
    """
    Normalized Maximum Eigengap based Spectral Clustering (NME-SC)
    uses Eigengap analysis to get an estimated p-value for
    affinity binarization and an estimated number of speakers.

    p_value (also referred to as p_neighbors) is for taking
    top p number of affinity values and convert those to 1 while
    convert the rest of values to 0.

    p_value can be also tuned on a development set without performing
    NME-analysis.

    Reference: Auto-Tuning Spectral Clustering for Speaker Diarization
    Using Normalized Maximum Eigengap (https://arxiv.org/abs/2003.02405)

    Parameters:
        Please refer to def __init__()

    Methods:
        _NMEanalysis():
            Performs NME-analysis to estimate p_value and the number of speakers.

        subsampleAffinityMat(NME_mat_size):
            Subsamples the number of speakers to reduce the computational load.

        getPvalueList():
            Generates a list contains p-values that need to be examined.

        getEigRatio(p_neighbors):
            calculates g_p, which is a ratio between p_neighbors and the maximum eigengap.

        getLamdaGaplist(lambdas):
            Calculates lambda gap values from an array contains ambda values.

        estimateNumofSpeakers(affinity_mat):
            Estimates the number of speakers using lambda gap list.

    """

    def __init__(
        self,
        mat,
        max_num_speaker: int=10,
        max_rp_threshold: float=0.250,
        sparse_search: bool=True,
        sparse_search_volume: int=30,
        use_subsampling_for_NME: bool=True,
        fixed_thres: float=0.0,
        cuda: bool=False,
        NME_mat_size: int=512,
        device: torch.device=torch.device('cpu')
    ):
        """
        Parameters:
            mat: (numpy.array)
                Cosine similarity matrix calculated from speaker embeddings.

            max_num_speaker: (int)
                Maximum number of speakers for estimating number of speakers.
                Shows stable performance under 20.

            max_rp_threshold: (float)
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.25.

            sparse_search: (bool)
                To increase the speed of parameter estimation, sparse_search=True
                limits the number of p_values we search.

            sparse_search_volume: (int)
                The number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                Lower than 20 might cause a poor parameter estimation.

            use_subsampling_for_NME: (bool)
                Use subsampling to reduce the calculational complexity.
                Default is True.

            fixed_thres: (float or None)
                A fixed threshould can be used instead of estimating the
                threshold with NME analysis. If fixed_thres is float,
                it skips NME analysis part.

            cuda: (bool)
                Use cuda for Eigen decomposition if cuda=True.

            NME_mat_size: (int)
                Targeted size of matrix for NME analysis.


        """
        self.max_num_speaker: int = max_num_speaker
        self.max_rp_threshold = max_rp_threshold
        self.use_subsampling_for_NME = use_subsampling_for_NME
        self.NME_mat_size: int = NME_mat_size
        self.sparse_search = sparse_search
        self.sparse_search_volume = sparse_search_volume
        self.fixed_thres: float= fixed_thres
        self.cuda: bool= cuda
        self.eps = 1e-10
        self.max_N = torch.tensor(0)
        self.mat = mat
        self.p_value_list: torch.Tensor = torch.tensor(0)
        self.device = device
        print(f'running NMESC on {self.device}..')   
    
    def NMEanalysis(self):
        """
        Subsample the input matrix to reduce the computational load.
        """
        if self.use_subsampling_for_NME:
            subsample_ratio = self.subsampleAffinityMat(self.NME_mat_size)
        else:
            subsample_ratio = torch.tensor(1)

        # Scans p_values and find a p_value that generates
        # the smallest g_p value.
        eig_ratio_list = []
        est_spk_n_dict: Dict[int, torch.Tensor] = {}
        self.p_value_list = self.getPvalueList()
        for p_value in self.p_value_list:
            est_num_of_spk, g_p = self.getEigRatio(p_value)
            est_spk_n_dict[p_value.item()] = est_num_of_spk
            eig_ratio_list.append(g_p)
        
        index_nn = torch.argmin(torch.tensor(eig_ratio_list))
        rp_p_value = self.p_value_list[index_nn]
        affinity_mat = getAffinityGraphMat(self.mat, rp_p_value)

        # Checks whether affinity graph is fully connected.
        # If not, it adds minimum number of connections to make it fully connected.
        if not isGraphFullyConnected(affinity_mat, device=self.device):
            affinity_mat, rp_p_value = getMinimumConnection(self.mat, self.max_N, self.p_value_list, device=self.device)

        p_hat_value = (subsample_ratio * rp_p_value).type(torch.int)
        est_num_of_spk = est_spk_n_dict[rp_p_value.item()]
        return est_num_of_spk, p_hat_value

    def subsampleAffinityMat(self, NME_mat_size: int):
        """
        Perform Subsampling of affinity matrix.
        This subsampling is for calculational complexity, not for performance.
        The smaller NME_mat_size is,
            - the bigger the chance of missing a speaker.
            - the faster p-value estimation speed (based on eigen decomposition).

        Recommended NME_mat_size is 250~750.
        However, if there are speakers who speak for very short period of time in the recording,
        this subsampling might make the system miss the underrepresented speaker.
        Use this with caution.

        Parameters:
            NME_mat_size: (int)
                Targeted matrix size

        Returns:
            subsample_ratio : (float)
                The ratio between NME_mat_size and the original matrix size

        """
        subsample_ratio = torch.max(torch.tensor(1), torch.tensor(self.mat.shape[0] / NME_mat_size)).type(torch.int)
        self.mat = self.mat[::subsample_ratio.item(), ::subsample_ratio.item()]
        return subsample_ratio
    
    def getEigRatio(self, p_neighbors: int):
        """
        For a given p_neighbors value,
        calculates g_p, which is a ratio
        between p_neighbors and the maximum eigengap.

        For more details: https://arxiv.org/abs/2003.02405

        Parameters:
            p_neighbors: (int)
                Determines how many binary graph connections we want to keep for each row.

        Returns:
            est_num_of_spk: (int)
                Estimated number of speakers

            g_p: (float)
                The ratio between p_neighbors value and the maximum eigen gap value.
        """
        affinity_mat = getAffinityGraphMat(self.mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = estimateNumofSpeakers(affinity_mat, self.max_num_speaker, self.cuda, device=self.device)
        arg_sorted_idx = torch.argsort(lambda_gap_list[: self.max_num_speaker], descending=True)
        max_key = arg_sorted_idx[0]
        max_eig_gap = lambda_gap_list[max_key] / (max(lambdas) + self.eps)
        g_p = (p_neighbors / self.mat.shape[0]) / (max_eig_gap + self.eps)

        return est_num_of_spk, g_p

    def getPvalueList(self):
        """
        Generates a p-value (p_neighbour) list for searching.
        """
        if self.fixed_thres > 0.0:
            p_value_list = torch.floor(torch.tensor(self.mat.shape[0] * self.fixed_thres)).type(torch.int)
            self.max_N = p_value_list[0]
        else:
            self.max_N = torch.floor(torch.tensor(self.mat.shape[0] * self.max_rp_threshold)).type(torch.int)
            if self.sparse_search:
                N = torch.min(self.max_N, torch.tensor(self.sparse_search_volume).type(torch.int))
                p_value_list = torch.linspace(start=1, end=self.max_N, steps=N).type(torch.int)
            else:
                p_value_list = torch.arange(1, self.max_N)

        return p_value_list

# @torch.jit.script
def COSclustering(
    uniq_embs_and_timestamps: List[List[List[torch.Tensor]]],
    oracle_num_speakers=None,
    max_num_speaker: int=8,
    min_samples_for_NMESC: int=6,
    enhanced_count_thres: int=80,
    max_rp_threshold: float=0.25,
    sparse_search_volume: int=30,
    fixed_thres: float=0.0,
    cuda=False,
    device: torch.device=torch.device('cpu'),
):
    """
    Clustering method for speaker diarization based on cosine similarity.

    Parameters:
        uniq_embs_and_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_embs_and_timestamps contains only one scale, single scale diarization
            is performed.

        oracle_num_speaker: (int or None)
            Oracle number of speakers if known else None

        max_num_speaker: (int)
            Maximum number of clusters to consider for each session

        min_samples_for_NMESC: (int)
            Minimum number of samples required for NME clustering, this avoids
            zero p_neighbour_lists. If the input has fewer segments than min_samples,
            it is directed to the enhanced speaker counting mode.

        enhanced_count_thres: (int)
            For short audio recordings under 60 seconds, clustering algorithm cannot
            accumulate enough amount of speaker profile for each cluster.
            Thus, getEnhancedSpeakerCount() employs anchor embeddings (dummy representations)
            to mitigate the effect of cluster sparsity.
            enhanced_count_thres = 80 is recommended.

        max_rp_threshold: (float)
            Limits the range of parameter search.
            Clustering performance can vary depending on this range.
            Default is 0.25.

        sparse_search_volume: (int)
            The number of p_values we search during NME analysis.
            Default is 30. The lower the value, the faster NME-analysis becomes.
            Lower than 20 might cause a poor parameter estimation.

        fixed_thres: (float)
            If fixed_thres value is provided, NME-analysis process will be skipped.
            This value should be optimized on a development set to obtain a quality result.
            Default is None and performs NME-analysis to estimate the threshold.

    Returns:
        Y: (List[int])
            Speaker label for each segment.
    """
    # Get base-scale embedding from uniq_embs_and_timestamps.
    uniq_scale_list = uniq_embs_and_timestamps[1]
    emb = uniq_scale_list[0][0]
    cuda, device= True, torch.device("cuda:0")
    # cuda, device= False, torch.device("cpu")
    if emb.shape[0] == 1:
        return torch.zeros((1,), dtype=torch.int32)
    else:
        est_num_of_spk_enhanced = None

    if oracle_num_speakers:
        max_num_speaker = oracle_num_speakers

    mat, emb = _getMultiScaleCosAffinityMatrix(uniq_embs_and_timestamps[0], uniq_embs_and_timestamps[1], device=device)
    emb = emb.to(device)
    
    nmesc = NMESC(
        mat,
        max_num_speaker=max_num_speaker,
        max_rp_threshold=max_rp_threshold,
        sparse_search=True,
        sparse_search_volume=sparse_search_volume,
        fixed_thres=fixed_thres,
        NME_mat_size=300,
        cuda=cuda,
        device=device,
    )

    if emb.shape[0] > min_samples_for_NMESC:
        est_num_of_spk, p_hat_value = nmesc.NMEanalysis()
        affinity_mat = getAffinityGraphMat(mat, p_hat_value)
    else:
        affinity_mat = mat

    if oracle_num_speakers:
        est_num_of_spk = oracle_num_speakers

    spectral_model = SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda, device=device)
    Y = spectral_model.predict(affinity_mat)
    return Y
