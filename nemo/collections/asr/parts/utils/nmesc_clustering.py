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

# NME-SC clustering is based on the implementation from the paper
# https://arxiv.org/pdf/2003.02405.pdf and the implementation from
# https://github.com/tango4j/Auto-Tuning-Spectral-Clustering.

from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import OneHotEncoder
from torch.linalg import eigh, eigvalsh


@torch.jit.script
def cos_similarity(a: torch.Tensor, b: torch.Tensor, eps=torch.tensor(3.5e-4)) -> torch.Tensor:
    """
    Calculate cosine similarities of the given two set of tensors. The output is an N by N
    matrix where N is the number of feature vectors.

    Args:
        a (Tensor):
            Matrix containing speaker representation vectors. (N x embedding_dim)
        b (Tensor):
            Matrix containing speaker representation vectors. (N x embedding_dim)

    Returns:
        res (Tensor):
            N by N matrix containing the cosine similarities of the values.
    """
    a_norm = a / (torch.norm(a, dim=1).unsqueeze(1) + eps)
    b_norm = b / (torch.norm(a, dim=1).unsqueeze(1) + eps)
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    res.fill_diagonal_(1)
    return res


@torch.jit.script
def ScalerMinMax(X: torch.Tensor) -> torch.Tensor:
    """
    Min-max scale the input affinity matrix X, which will lead to a dynamic range of [0, 1].

    Args:
        X (Tensor):
            Matrix containing cosine similarity values among embedding vectors (N x N)

    Returns:
        v_norm (Tensor):
            Min-max normalized value of X.
    """
    v_min, v_max = X.min(), X.max()
    v_norm = (X - v_min) / (v_max - v_min)
    return v_norm


@torch.jit.script
def getEuclideanDistance(
    specEmbA: torch.Tensor, specEmbB: torch.Tensor, device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Calculate Euclidean distances from the given feature tensors.

    Args:
        specEmbA (Tensor):
            Matrix containing spectral embedding vectors from eigenvalue decomposition (N x embedding_dim).
        specEmbB (Tensor):
            Matrix containing spectral embedding vectors from eigenvalue decomposition (N x embedding_dim).

    Returns:
        dis (Tensor):
            Euclidean distance values of the two sets of spectral embedding vectors.
    """
    specEmbA, specEmbB = specEmbA.to(device), specEmbB.to(device)
    A, B = specEmbA.unsqueeze(dim=1), specEmbB.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis


@torch.jit.script
def kmeans_plusplus_torch(
    X: torch.Tensor,
    n_clusters: int,
    random_state: int,
    n_local_trials: int = 30,
    device: torch.device = torch.device('cpu'),
):
    """
    Choose initial centroids for initializing k-means algorithm. The performance of
    k-means algorithm can vary significantly by the initial centroids. To alleviate
    this problem, k-means++ algorithm chooses initial centroids based on the probability
    proportional to the distance from the formally chosen centroids. The centroids
    selected by k-means++ algorithm improve the chance of getting more accurate and
    stable clustering results. The overall implementation of k-means++ algorithm is
    inspired by the numpy based k-means++ implementation in:
        https://github.com/scikit-learn/scikit-learn

    Originally, the implementation of the k-means++ algorithm in scikit-learn is based
    on the following research article:
        Arthur, David, and Sergei Vassilvitskii. k-means++: The advantages of careful
        seeding. Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
        algorithms, Society for Industrial and Applied Mathematics (2007)

    Args:
        X (Tensor):
            Matrix containing cosine similarity values among embedding vectors (N x N)
        n_clusters (int):
            Maximum number of speakers for estimating number of speakers.
            Shows stable performance under 20.
        random_state (int):
            Seed variable for setting up a random state.
        n_local_trials (int):
            Number of trials for creating initial values of the center points.
        device (torch.device)
            Torch device variable.

    Returns:
        centers (Tensor):
            The coordinates for center points that are used for initializing k-means algorithm.
        indices (Tensor):
            The indices of the best candidate center points.
    """
    torch.manual_seed(random_state)
    X = X.to(device)
    n_samples, n_features = X.shape

    centers = torch.zeros(n_clusters, n_features, dtype=X.dtype)
    center_id = torch.randint(0, n_samples, (1,)).long()
    indices = torch.full([n_clusters,], -1, dtype=torch.int)

    centers[0] = X[center_id].squeeze(0)
    indices[0] = center_id.squeeze(0)

    centers = centers.to(device)
    closest_dist_diff = centers[0, None].repeat(1, X.shape[0]).view(X.shape[0], -1) - X
    closest_dist_sq = closest_dist_diff.pow(2).sum(dim=1).unsqueeze(dim=0)
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = torch.rand(n_local_trials) * current_pot.item()

        if len(closest_dist_sq.shape) > 1:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=1)[0]
        else:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=0)

        candidate_ids = torch.searchsorted(torch_cumsum, rand_vals.to(device))

        N_ci = candidate_ids.shape[0]
        distance_diff = X[candidate_ids].repeat(1, X.shape[0]).view(X.shape[0] * N_ci, -1) - X.repeat(N_ci, 1)
        distance = distance_diff.pow(2).sum(dim=1).view(N_ci, -1)
        distance_to_candidates = torch.minimum(closest_dist_sq, distance)
        candidates_pot = distance_to_candidates.sum(dim=1)

        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return centers, indices


@torch.jit.script
def kmeans_torch(
    X: torch.Tensor,
    num_clusters: int,
    threshold: float = 1e-4,
    iter_limit: int = 15,
    random_state: int = 0,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Run k-means algorithm on the given set of spectral embeddings in X. The threshold
    and iter_limit variables are set to show the best performance on speaker diarization
    tasks. The overall implementation of k-means algorithm is inspired by the k-means
    algorithm implemented in https://github.com/scikit-learn/scikit-learn.

    References:
        Arthur, David, and Sergei Vassilvitskii. k-means++: The advantages of careful
        seeding. Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
        algorithms, Society for Industrial and Applied Mathematics (2007).

    Args:
        X (Tensor):
            Cosine similarity matrix calculated from speaker embeddings
        num_clusters (int):
            The estimated number of speakers.
        threshold (float):
            This threshold limits the change of center values. If the square of
            the center shift values are bigger than this threshold, the iteration stops.
        iter_limit (int):
            The maximum number of iterations that is allowed by the k-means algorithm.
        device (torch.device):
            Torch device variable

    Returns:
        selected_cluster_indices (Tensor):
            The assigned cluster labels from the k-means clustering.
    """
    # Convert tensor type to float
    X = X.float().to(device)
    input_size = X.shape[0]

    # Initialize the cluster centers with kmeans_plusplus algorithm.
    plusplus_init_states = kmeans_plusplus_torch(X, n_clusters=num_clusters, random_state=random_state, device=device)
    centers = plusplus_init_states[0]

    iter_count = 0
    selected_cluster_indices = torch.zeros(input_size).long()

    for iter_count in range(iter_limit):
        euc_dist = getEuclideanDistance(X, centers, device=device)

        if len(euc_dist.shape) <= 1:
            break
        else:
            selected_cluster_indices = torch.argmin(euc_dist, dim=1)

        center_inits = centers.clone()

        for index in range(num_clusters):
            selected_cluster = torch.nonzero(selected_cluster_indices == index).squeeze().to(device)
            chosen_indices = torch.index_select(X, 0, selected_cluster)

            if chosen_indices.shape[0] == 0:
                chosen_indices = X[torch.randint(len(X), (1,))]

            centers[index] = chosen_indices.mean(dim=0)

        # Calculate the delta from center_inits to centers
        center_delta_pow = torch.pow((centers - center_inits), 2)
        center_shift_pow = torch.pow(torch.sum(torch.sqrt(torch.sum(center_delta_pow, dim=1))), 2)

        # If the cluster centers are not changing significantly, stop the loop.
        if center_shift_pow < threshold:
            break

    return selected_cluster_indices


@torch.jit.script
def getTheLargestComponent(affinity_mat: torch.Tensor, seg_index: int, device: torch.device) -> torch.Tensor:
    """
    Find the largest affinity_mat connected components for each given node.
    This is for checking whether the affinity_mat is fully connected.

    Args:
        affinity_mat (Tensor):
            A square matrix (tensor) containing normalized cosine distance values
        seg_index (int):
            The segment index that is targeted to be explored.

    Returns:
        connected_nodes (Tensor):
            A tensor containing booleans that indicate whether the node is connected.

    """
    num_of_segments = affinity_mat.shape[0]

    connected_nodes = torch.zeros(num_of_segments, dtype=torch.bool).to(device)
    nodes_to_explore = torch.zeros(num_of_segments, dtype=torch.bool).to(device)

    nodes_to_explore[seg_index] = True
    nodes_to_explore = nodes_to_explore.to(device)
    for k in range(num_of_segments):
        last_num_component = connected_nodes.sum()
        torch.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break

        indices = (nodes_to_explore == torch.tensor(True)).nonzero().t().squeeze()
        if len(indices.size()) == 0:
            indices = indices.unsqueeze(0)
        for i in indices:
            neighbors = affinity_mat[i].to(device)
            torch.logical_or(nodes_to_explore, neighbors.squeeze(0), out=nodes_to_explore)
    return connected_nodes


@torch.jit.script
def isGraphFullyConnected(affinity_mat: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Check whether the given affinity matrix is a fully connected graph.
    """
    return getTheLargestComponent(affinity_mat, 0, device).sum() == affinity_mat.shape[0]


@torch.jit.script
def getKneighborsConnections(affinity_mat: torch.Tensor, p_value: int) -> torch.Tensor:
    """
    Binarize top-p values for each row from the given affinity matrix.
    """
    binarized_affinity_mat = torch.zeros_like(affinity_mat).int()
    for i in range(affinity_mat.shape[0]):
        line = affinity_mat[i, :]
        sorted_idx = torch.argsort(line, descending=True)
        indices = sorted_idx[:p_value]
        binarized_affinity_mat[indices, i] = torch.ones(indices.shape[0]).to(affinity_mat.device).int()
    return binarized_affinity_mat


@torch.jit.script
def getAffinityGraphMat(affinity_mat_raw: torch.Tensor, p_value: int) -> torch.Tensor:
    """
    Calculate a binarized graph matrix and
    symmetrize the binarized graph matrix.
    """
    if p_value <= 0:
        X = affinity_mat_raw
    else:
        X = getKneighborsConnections(affinity_mat_raw, p_value)
    symm_affinity_mat = 0.5 * (X + X.T)
    return symm_affinity_mat


@torch.jit.script
def getMinimumConnection(
    mat: torch.Tensor, max_N: torch.Tensor, n_list: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate connections until fully connect all the nodes in the graph.
    If the graph is not fully connected, it might generate inaccurate results.
    """
    p_value = torch.tensor(1)
    affinity_mat = getAffinityGraphMat(mat, p_value)
    for i, p_value in enumerate(n_list):
        fully_connected = isGraphFullyConnected(affinity_mat, device)
        affinity_mat = getAffinityGraphMat(mat, p_value)
        if fully_connected or p_value > max_N:
            break

    return affinity_mat, p_value


@torch.jit.script
def getRepeatedList(mapping_argmat: torch.Tensor, score_mat_size: torch.Tensor) -> torch.Tensor:
    """
    Count the numbers in the mapping dictionary and create lists that contain
    repeated indices that will be used for creating a repeated affinity matrix.
    This repeated matrix is then used for fusing multiple affinity values.
    """
    repeat_list = torch.zeros(score_mat_size, dtype=torch.int32).to(mapping_argmat.device)
    idxs, counts = torch.unique(mapping_argmat, return_counts=True)
    repeat_list[idxs] = counts.int().to(mapping_argmat.device)
    return repeat_list


@torch.jit.script
def get_argmin_mat(timestamps_in_scales: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Calculate the mapping between the base scale and other scales. A segment from a longer scale is
    repeatedly mapped to a segment from a shorter scale or the base scale.

    Args:
        timestamps_in_scales (list):
            List containing timestamp tensors for each scale.
            Each tensor has dimensions of (Number of base segments) x 2.

    Returns:
        session_scale_mapping_list (list):
            List containing argmin arrays indexed by scale index.
    """
    scale_list = list(range(len(timestamps_in_scales)))
    segment_anchor_list = []
    for scale_idx in scale_list:
        time_stamps_float = timestamps_in_scales[scale_idx]
        segment_anchor_list.append(torch.mean(time_stamps_float, dim=1))

    base_scale_idx = max(scale_list)
    base_scale_anchor = segment_anchor_list[base_scale_idx]
    session_scale_mapping_list = []
    for scale_idx in scale_list:
        curr_scale_anchor = segment_anchor_list[scale_idx]
        curr_mat = torch.tile(curr_scale_anchor, (base_scale_anchor.shape[0], 1))
        base_mat = torch.tile(base_scale_anchor, (curr_scale_anchor.shape[0], 1)).t()
        argmin_mat = torch.argmin(torch.abs(curr_mat - base_mat), dim=1)
        session_scale_mapping_list.append(argmin_mat)
    return session_scale_mapping_list


@torch.jit.script
def getCosAffinityMatrix(emb: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity values among speaker embeddings then min-max normalize
    the affinity matrix.

    Args:
        emb (Tensor):
            Matrix containing embedding vectors. emb variable should be float(FP32) type to make the data-type
            compatible with torch.mm operation for both CPU and GPU(CUDA).
            dimension: (Number of embedding vectors) x (embedding dimension)

    Returns:
        sim_d (Tensor):
            Matrix containing cosine similarity values among the given embedding vectors.
            dimension: (Number of embedding vectors) x (Number of embedding vectors)
    """
    emb = emb.float()
    sim_d = cos_similarity(emb, emb)
    sim_d = ScalerMinMax(sim_d)
    return sim_d

@torch.jit.script
def getTempInterpolMultiScaleCosAffinityMatrix(
    multiscale_weights: torch.Tensor,
    embeddings_in_scales: List[torch.Tensor],
    timestamps_in_scales: List[torch.Tensor],
    device: torch.device = torch.device('cpu'),
):
    """
    Calculate cosine similarity values among speaker embeddings for each scale then
    apply multiscale weights to calculate the fused similarity matrix.

    Take the embedding from the scale (scale_interpolation_index-1)-th scale and find the indexes that are
    in the range of [-half_scale, half_scale]. The distance to the center of the base-scale is used for
    calculating the interpolation weight. The distance is squared then normalized to create an interpolation
    weight vector interpol_w. Using the interpolation weight interpol_w and target embedding indexes target_bool,
    interpolated embedding is calculated.

    Args:
        uniq_embs_and_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_embs_and_timestamps contains only one scale, single scale diarization
            is performed.

    Returns:
        fused_sim_d (torch.tensor):
            This function generates an affinity matrix that is obtained by calculating
            the weighted sum of the affinity matrices from the different scales.
        base_scale_emb (torch.tensor):
            The base scale embedding (the embeddings from the finest scale)
    """
    rep_mat_list = []
    multiscale_weights = multiscale_weights.to(device)
    session_scale_mapping_dict = get_argmin_mat(timestamps_in_scales)
    scale_list = list(range(len(timestamps_in_scales)))
    for scale_idx in scale_list:
        mapping_argmat = session_scale_mapping_dict[scale_idx]
        emb_t = embeddings_in_scales[scale_idx].half().to(device)
        mapping_argmat = mapping_argmat.to(device)
        repeat_list = getRepeatedList(mapping_argmat, torch.tensor(emb_t.shape[0])).to(device)
        rep_emb_t = torch.repeat_interleave(emb_t, repeats=repeat_list, dim=0)
        rep_mat_list.append(rep_emb_t)
    stacked_scale_embs = torch.stack(rep_mat_list)
    context_emb = torch.matmul(stacked_scale_embs.permute(2, 1, 0), multiscale_weights.t().half()).squeeze().t()
    if len(context_emb.shape) < 2:
        context_emb = context_emb.unsqueeze(0)
    context_emb =context_emb.to(device)
    return context_emb, session_scale_mapping_dict


@torch.jit.script
def getMultiScaleCosAffinityMatrix(
    multiscale_weights: torch.Tensor,
    embeddings_in_scales: List[torch.Tensor],
    timestamps_in_scales: List[torch.Tensor],
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Calculate cosine similarity values among speaker embeddings for each scale then
    apply multiscale weights to calculate the fused similarity matrix.

    Args:
        uniq_embs_and_timestamps (dict):
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_embs_and_timestamps contains only one scale, single scale diarization
            is performed.

    Returns:
        fused_sim_d (Tensor):
            This function generates an affinity matrix that is obtained by calculating
            the weighted sum of the affinity matrices from the different scales.
    """
    multiscale_weights = multiscale_weights.to(device)
    score_mat_list, repeated_tensor_list = [], []
    session_scale_mapping_list = get_argmin_mat(timestamps_in_scales)
    scale_list = list(range(len(timestamps_in_scales)))
    for scale_idx in scale_list:
        mapping_argmat = session_scale_mapping_list[scale_idx]
        emb_t = embeddings_in_scales[scale_idx].half().to(device)
        score_mat_torch = getCosAffinityMatrix(emb_t)
        repeat_list = getRepeatedList(mapping_argmat, torch.tensor(score_mat_torch.shape[0])).to(device)
        repeated_tensor_0 = torch.repeat_interleave(score_mat_torch, repeats=repeat_list, dim=0)
        repeated_tensor_1 = torch.repeat_interleave(repeated_tensor_0, repeats=repeat_list, dim=1)
        repeated_tensor_list.append(repeated_tensor_1)
    repp = torch.stack(repeated_tensor_list).float()
    fused_sim_d = torch.matmul(repp.permute(2, 1, 0), multiscale_weights.t()).squeeze(2).t()
    return fused_sim_d


@torch.jit.script
def getLaplacian(X: torch.Tensor) -> torch.Tensor:
    """
    Calculate a laplacian matrix from an affinity matrix X.
    """
    X.fill_diagonal_(0)
    D = torch.sum(torch.abs(X), dim=1)
    D = torch.diag_embed(D)
    L = D - X
    return L


@torch.jit.script
def eigDecompose(
    laplacian: torch.Tensor, cuda: bool, device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate eigenvalues and eigenvectors from the Laplacian matrix.
    """
    if cuda:
        if device is None:
            device = torch.cuda.current_device()
        laplacian = laplacian.float().to(device)
    else:
        laplacian = laplacian.float()
    lambdas, diffusion_map = eigh(laplacian)
    return lambdas, diffusion_map


@torch.jit.script
def eigValueSh(laplacian: torch.Tensor, cuda: bool, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Calculate only eigenvalues from the Laplacian matrix.
    """
    if cuda:
        if device is None:
            device = torch.cuda.current_device()
        laplacian = laplacian.float().to(device)
    else:
        laplacian = laplacian.float()
    lambdas = eigvalsh(laplacian)
    return lambdas


@torch.jit.script
def getLamdaGaplist(lambdas: torch.Tensor) -> torch.Tensor:
    """
    Calculate the gaps between lambda values.
    """
    if torch.is_complex(lambdas):
        lambdas = torch.real(lambdas)
    return lambdas[1:] - lambdas[:-1]


@torch.jit.script
def addAnchorEmb(emb: torch.Tensor, anchor_sample_n: int, anchor_spk_n: int, sigma: float) -> torch.Tensor:
    """
    Add randomly generated synthetic embeddings to make eigenanalysis more stable.
    We refer to these embeddings as anchor embeddings.

    emb (Tensor):
        The input embedding from the embedding extractor.
    anchor_sample_n (int):
        Number of embedding samples per speaker.
        anchor_sample_n = 10 is recommended.
    anchor_spk_n (int):
        Number of speakers for synthetic embedding.
        anchor_spk_n = 3 is recommended.
    sigma (int):
        The amplitude of synthetic noise for each embedding vector.
        If the sigma value is too small, under-counting could happen.
        If the sigma value is too large, over-counting could happen.
        sigma = 50 is recommended.
    """
    emb_dim = emb.shape[1]
    std_org = torch.std(emb, dim=0)
    sigma = torch.tensor(sigma).to(emb.device)
    new_emb_list = []
    for _ in range(anchor_spk_n):
        emb_m = torch.tile(torch.randn(1, emb_dim), (anchor_sample_n, 1)).half().to(emb.device)
        emb_noise = torch.randn(anchor_sample_n, emb_dim).T.to(emb.device).half()
        emb_noise = torch.matmul(
            torch.diag(std_org).half(), emb_noise / torch.max(torch.abs(emb_noise), dim=0)[0].unsqueeze(0)
        ).T
        emb_gen = emb_m + sigma * emb_noise
        new_emb_list.append(emb_gen)

    new_emb_list.append(emb)
    new_emb_np = torch.vstack(new_emb_list)
    return new_emb_np


def getEnhancedSpeakerCount(
    emb: torch.Tensor,
    random_test_count: int = 5,
    anchor_spk_n: int = 3,
    anchor_sample_n: int = 10,
    sigma: float = 50,
    cuda: bool = False,
) -> torch.Tensor:
    """
    Calculate the number of speakers using NME analysis with anchor embeddings. Add dummy speaker
    embedding vectors and run speaker counting multiple times to enhance the speaker counting accuracy
    for the short audio samples.

    Args:
        emb (Tensor):
            The input embedding from the embedding extractor.
        cuda (bool):
            Use cuda for the operations if cuda==True.
        random_test_count (int):
            Number of trials of the enhanced counting with randomness.
            The higher the count, the more accurate the enhanced counting is.
        anchor_spk_n (int):
            Number of speakers for synthetic embedding.
            anchor_spk_n = 3 is recommended.
        anchor_sample_n (int):
            Number of embedding samples per speaker.
            anchor_sample_n = 10 is recommended.
        sigma (float):
            The amplitude of synthetic noise for each embedding vector.
            If the sigma value is too small, under-counting could happen.
            If the sigma value is too large, over-counting could happen.
            sigma = 50 is recommended.

    Returns:
        comp_est_num_of_spk (Tensor):
            The estimated number of speakers. `anchor_spk_n` is subtracted from the estimated
            number of speakers to factor out the dummy speaker embedding vectors.
    """
    est_num_of_spk_list: List[int] = []
    for seed in range(random_test_count):
        torch.manual_seed(seed)
        emb_aug = addAnchorEmb(emb, anchor_sample_n, anchor_spk_n, sigma)
        mat = getCosAffinityMatrix(emb_aug)
        nmesc = NMESC(
            mat,
            max_num_speakers=emb.shape[0],
            max_rp_threshold=0.15,
            sparse_search=True,
            sparse_search_volume=10,
            fixed_thres=-1.0,
            nme_mat_size=300,
            cuda=cuda,
        )
        est_num_of_spk, _ = nmesc.forward()
        est_num_of_spk_list.append(est_num_of_spk.item())
    comp_est_num_of_spk = torch.mode(torch.tensor(est_num_of_spk_list))[0] - anchor_spk_n
    return comp_est_num_of_spk


@torch.jit.script
def split_input_data(
    embeddings_in_scales: torch.Tensor,
    timestamps_in_scales: torch.Tensor,
    multiscale_segment_counts: torch.LongTensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Split multiscale embeddings and multiscale timestamps and put split scale-wise data into python lists.
    This formatting function is needed to make the input type as `torch.Tensor`.

    Args:
        embeddings_in_scales (Tensor):
            Concatenated Torch tensor containing embeddings in multiple scales
        timestamps_in_scales (Tensor):
            Concatenated Torch tensor containing timestamps in multiple scales
        multiscale_segment_counts (LongTensor):
            Concatenated Torch LongTensor containing number of segments per each scale

    Returns:
        embeddings_in_scales (list):
            List containing split embedding tensors by each scale
        timestamps_in_scales (list):
            List containing split timestamps tensors by each scale
    """
    split_index: List[int] = multiscale_segment_counts.tolist()
    embeddings_in_scales = torch.split(embeddings_in_scales, split_index, dim=0)
    timestamps_in_scales = torch.split(timestamps_in_scales, split_index, dim=0)
    embeddings_in_scales, timestamps_in_scales = list(embeddings_in_scales), list(timestamps_in_scales)
    return embeddings_in_scales, timestamps_in_scales


@torch.jit.script
def estimateNumofSpeakers(
    affinity_mat: torch.Tensor, max_num_speakers: int, cuda: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate the number of speakers using eigendecomposition on the Laplacian Matrix.

    Args:
        affinity_mat (Tensor):
            N by N affinity matrix
        max_num_speakers (int):
            Maximum number of clusters to consider for each session
        cuda (bool):
            If cuda available eigendecomposition is computed on GPUs.

    Returns:
        num_of_spk (Tensor):
            The estimated number of speakers
        lambdas (Tensor):
            The lambda values from eigendecomposition
        lambda_gap (Tensor):
            The gap between the lambda values from eigendecomposition
    """
    with torch.no_grad():
        laplacian = getLaplacian(affinity_mat)
        lambdas = eigValueSh(laplacian, cuda=cuda)
        lambdas = torch.sort(lambdas)[0]
        lambda_gap = getLamdaGaplist(lambdas)
        num_of_spk = torch.argmax(lambda_gap[: min(max_num_speakers, lambda_gap.shape[0])]) + 1
    return num_of_spk, lambdas, lambda_gap


def hungarian_algorithm(
    spk_count: int, 
    U_set: List[int], 
    cmm_P: torch.Tensor, 
    cmm_Q: torch.Tensor, 
    PmQ: List[int], 
    QmP: List[int]
    ) -> np.array:
    """
    Find a mapping that minimizes the matching cost between the label P and Q.
    One-hot encodding is employed to represent sequence and calculate the cost.

    Args:
        spk_count (int):
            Estimated speaker count
        U_set (list):
            Whole set of the estimated speakers
        cmm_P (Tensor):
            Length-matched old sequence
        cmm_Q (Tensor):
            Length-matched new sequence
        PmQ (list):
            Set P - Q (Difference of sets)
        QmP (list):
            Set Q - P (Difference of sets)

    Returns:
        mapping_array (np.array):
            Mapped labels that minimizes the cost
    """
    enc = OneHotEncoder(handle_unknown='ignore')
    all_spks_labels = [[x] for x in range(len(U_set))]
    enc.fit(all_spks_labels)
    enc_P = enc.transform(cmm_P.reshape(-1, 1)).toarray()
    enc_Q = enc.transform(cmm_Q.reshape(-1, 1)).toarray()
    stacked = np.hstack((enc_P, enc_Q))
    cost = -1 * linear_kernel(stacked.T)[spk_count:, :spk_count]
    row_ind, col_ind = linear_sum_assignment(cost)

    # If number of are speakers in each vector is not the same
    mapping_array = np.arange(len(U_set)).astype(int)
    for x in range(col_ind.shape[0]):
        if x in (set(PmQ) | set(QmP)):
            mapping_array[x] = x
        else:
            mapping_array[x] = col_ind[x]
    return mapping_array


def get_minimal_indices(Y_new: torch.LongTensor) -> torch.LongTensor:
    """
    Force the unique indices of the labels to use the lowest numbers.

    Example:
        >>> Y_new = [3, 3, 3, 4, 4, 5]
        >>> get_minimal_indices(Y_new)
        Return:
            [0, 0, 0, 1, 1, 2]

    Args:
        Y_new (Tensor):
            Tensor containing cluster labels

    Returns:
        (Tensor): Newly mapped cluster labels that has minimized indicies
    """
    Y_new_enlisted = torch.unique(Y_new).sort()[0].to(torch.long)
    sequence = torch.arange(torch.max(Y_new_enlisted) + 1)
    sequence[Y_new_enlisted] = torch.arange(len(Y_new_enlisted))
    return sequence[Y_new]


def stitch_cluster_labels(Y_old: torch.Tensor, Y_new: torch.Tensor, with_history=True):
    """
    Run Hungarian algorithm (linear sum assignment) to find the best permutation mapping between
    the cumulated labels in history and the new clustering output labels.


    Args:
        Y_old (Tensor):
            Cumulated diarization labels. This will be concatenated with history embedding speaker label
            then compared with the predicted label Y_new.
        Y_new (Tensor):
            Contains predicted labels for reduced history embeddings concatenated with the predicted label.
            Permutation is not matched yet.

    Returns:
        mapping_array[Y] (Tensor):
            An output numpy array where the input Y_new is mapped with mapping_array.
    """
    Y_new = get_minimal_indices(Y_new)

    # TODO: This function needs to be converted to a fully torch.jit.script-able function.
    # For th
    Y_old = Y_old.cpu().numpy()
    Y_new = Y_new.cpu().numpy()

    if len(Y_old) == 0:
        matched_output = Y_new
    else:
        spk_count = max(len(set(Y_old)), len(set(Y_new)))
        P_raw, Q_raw = Y_old.astype(int), Y_new.astype(int)
        U_set = set(P_raw) | set(Q_raw)
        min_len = min(P_raw.shape[0], Q_raw.shape[0])
        P, Q = P_raw[:min_len], Q_raw[:min_len]
        PmQ, QmP = set(P) - set(Q), set(Q) - set(P)

        # P and Q occasionally have no common labels which means totally flipped (0<->1) labels.
        # This should be differentiated from the second case.
        if len(U_set) == 1:
            # When two speaker vectors are exactly the same: No need to encode.
            mapping_array = np.array([0, 0])
        else:
            # Run Hungarian algorithm if there are more than one speaker in universal set U.
            mapping_array = hungarian_algorithm(spk_count, U_set, P, Q, PmQ, QmP)
        matched_output = mapping_array[Y_new]
    matched_output = torch.tensor(matched_output)
    matched_output = get_minimal_indices(matched_output)
    return matched_output


@torch.jit.script
def calculate_removable_counts(
    removable_counts_mat: torch.Tensor,
    remain_count: int,
    num_clus: int,
    ) -> torch.Tensor:
    """
    Calculate removable counts based on the arguments and calculate how many counts should be
    removed from the each cluster. This function has `O(N)` (N = num_clus) time complexity to
    return the desired `removable_counts_mat`.

    Example:

        The original input to `get_merge_quantity` function:
        >>> pre_clus_labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        >>> num_to_be_removed = 3
        >>> min_count_per_cluster = 2

        Histogram: (`min_count_per_cluster`=2 is removed)
        0 |*****
        1 |***
        2 |*

        Inputs:
            >>> removable_counts_mat = [5, 3, 1]
            >>> remain_count = 6
            >>> num_clus = 3
        
        Interim results:
            >>> diff_counts 
            [1, 2, 2]
            >>> gradual_counts
            [3, 4, 2]
            >>> cumsum_counts
            [3, 7, 9]

        Return:
            >>> removable_counts_mat 
            [2, 1, 0]

    Args:
        removable_counts_mat (Tensor):
            Tensor containing how many vectors could be removed from each cluster
        remain_count (int):
            Integer value that indicates the number of vectors removed from the total set
        num_clus (int):
            Number of clusters in the given label sequence (cardinality of a label set)

    Returns:
        removable_counts_mat (Tensor):
            Tensor containing the number of vectors should be removed from each cluster
    """
    device = removable_counts_mat.device
    zero_padded_counts = torch.cat([torch.tensor([0]).to(device), 
                                    removable_counts_mat.sort()[0], 
                                    torch.tensor([0]).to(device)], 
                                    dim=0)
    removable_count_args = removable_counts_mat.sort(descending=True)[1]
    
    # Calculate the size difference between clusters
    diff_counts  = (zero_padded_counts[1:]- zero_padded_counts[-1])[:num_clus]
    gradual_counts = torch.arange(num_clus, 0, -1).to(device) * diff_counts
    cumsum_counts = torch.cumsum(gradual_counts, dim=0)
    count, remain_count_rem = 0, remain_count

    # Find how many remaining counts we can use
    ind: int = 0
    for ind, num in enumerate(cumsum_counts):
        if remain_count < num:
            break

    # Subtract the common values step by step
    if ind > 0:
        for knd in range(ind):
            removable_counts_mat[removable_count_args[:num_clus-knd]] -=  diff_counts[knd] 
            remain_count_rem -= int(diff_counts[knd].item())*(num_clus-knd)
    assert remain_count >= 0, "remain_count should never be negative."

    # Add remaining values
    num_labels = remain_count_rem // (num_clus - ind)
    rem_labels = remain_count_rem  % (num_clus - ind)
    removable_counts_mat[removable_count_args[:(num_clus - ind)]]-= num_labels
    removable_counts_mat[removable_count_args[:rem_labels]] -= 1
    return removable_counts_mat


@torch.jit.script
def get_merge_quantity(
    num_to_be_removed: int, 
    pre_clus_labels: torch.Tensor, 
    min_count_per_cluster: int,
) -> torch.Tensor:
    """
    Determine which embeddings we need to reduce or merge in history buffer.
    We want to merge or remove the embedding in the bigger cluster first.
    At the same time, we keep the minimum number of embedding per cluster
    with the variable named min_count_per_cluster.

    Constraint:
        - Each cluster should keep the number of vectors over `min_count_per_cluster`.
        - In total, `num_to_be_removed` of vectors should be removed from the total buffer.
        - While merging embeddings, minimize the gap between quantities between clusters.

    Example:
        >>> num_to_be_removed = 3
        >>> pre_clus_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        >>> min_count_per_cluster = 2
        >>> get_merge_quantity(num_to_be_removed, pre_clus_labels, min_count_per_cluster)
        Return:   
            torch.tensor([2, 1, 0]) 
        >>> # Sum should be equal to `num_to_be_removed` which is 3

    Args:
        num_to_be_removed: (int)
            the quantity of the newly obtained embedding from the new stream of input.
        pre_clus_labels: (Tensor)
            the speaker labels of (the history_embedding_buffer_emb) + (the new embeddings to be added)
        min_count_per_cluster: (int)
            Minimum vector quantity for each cluster

    Returns:
        removable_counts_mat: (Tensor)
            Tensor containing the number of vectors should be removed from each cluster
    """
    if num_to_be_removed > pre_clus_labels.shape[0]-1:
        raise ValueError(f"num_to_be_removed: {num_to_be_removed} should be less than pre_clus_labels length - 1 {pre_clus_labels.shape[0]-1}")
    remain_count = pre_clus_labels.shape[0] - num_to_be_removed
    spk_freq_count = torch.bincount(pre_clus_labels)
    num_clus = len(torch.unique(pre_clus_labels))
    if remain_count < min_count_per_cluster * num_clus:
        raise ValueError(f"The remaining embedding vectors should be more than { min_count_per_cluster * num_clus }")
    
    # Minimum vector counts should be excluded from the removable amount
    min_seg_count = torch.tensor([min_count_per_cluster]*len(spk_freq_count)).to(pre_clus_labels.device)
    min_seg_count_mat = torch.stack((min_seg_count, spk_freq_count)).min(0)[0]

    # Exclude minimum quantities from the removable count matrix
    remain_count -= int(torch.sum(min_seg_count_mat))
    removable_counts_mat = spk_freq_count - min_seg_count_mat
    
    # Calculate removable counts from `remain_count` variable
    removable_counts_mat = calculate_removable_counts(removable_counts_mat, remain_count, num_clus) 
    if int(removable_counts_mat.sum()) != num_to_be_removed:
        raise ValueError("Sum of `removable_counts_mat` is not equal to `num_to_be_removed` variable.")
    if not torch.all(removable_counts_mat >= 0):
        raise ValueError("Every value in `removable_counts_mat` should be non-negative value.")
    return removable_counts_mat


@torch.jit.script
def merge_vectors(
    selected_inds: torch.Tensor, 
    emb_ndx: torch.Tensor, 
    pre_cluster_labels: torch.Tensor
    ):
    """
    Merge feature (embedding) vectors estimated to be the same cluster label.

    Args:
        selected_inds (Tensor):
            Selected indices for merging
        emb_ndx (Tensor):
            Feature (embedding) vectors
            Dimension: (original vector counts) x (feature dimension)
        pre_cluster_labels (Tensor):
            Original cluster labels before merging

    Returns:
        merged_vecs (Tensor):
            Merged feature vectors that are concatenated
            Dimension: (merged vector counts) x (feature dimension)
        merged_clus_labels (Tensor):
            Cluster labels for the merged feature vectors
            Dimension: (merged vector counts)
        index_mapping (Tuple):
            index_mapping[0] contains bypassed vector labels
            index_mapping[1] contains merged  vector labels
    """
    if emb_ndx.shape[0] != pre_cluster_labels.shape[0]:
        raise ValueError("pre_cluster_labels and emb_ndx have mismatch in dimension")
    avg_emb = torch.mean(emb_ndx[selected_inds, :], dim=0)
    merged_clus_labels = pre_cluster_labels[selected_inds]
    selected_inds_list: List[int] = selected_inds.tolist()
    bypass_inds_list: List[int] = []
    for k in range(emb_ndx.shape[0]):
        if k not in selected_inds_list:
            bypass_inds_list.append(k)
    bypass_inds = torch.tensor(bypass_inds_list)
    selected_inds = torch.tensor(selected_inds_list)
    merged_vecs = torch.vstack((emb_ndx[bypass_inds], avg_emb))
    merged_clus_labels = torch.hstack((pre_cluster_labels[bypass_inds], merged_clus_labels[0]))
    index_mapping: Tuple[torch.Tensor, torch.Tensor] = (bypass_inds, selected_inds)
    return merged_vecs, merged_clus_labels, index_mapping


@torch.jit.script
def get_closest_embeddings(
    label_aff_mat: torch.Tensor, 
    target_emb_index: torch.Tensor, 
    merge_quantity: int
    ) -> torch.Tensor:
    """
    Get the indeces of the embedding vectors we want to merge.
    Example:
        >>> label_aff_mat = [[1.0, 0.2, 0.8],
                             [0.2, 1.0, 0.4],
                             [0.8, 0.4, 1.0]]
        >>> affinity_mat.sum(0) 
        [2.0, 1.6, 2.2]
        # The closest embedding vectors are index 0, 2.

    Args:
        label_aff_mat: (Tensor)
            Symmetric affinity matrix of the given embedding vector set.
        target_emb_index: (Tensor)
            Targeted speaker index
        merge_quantity: (int)
            The count of t

    Output:
        index_2d: (numpy.array)
    """
    comb_limit = int(target_emb_index.shape[0]-1)
    if merge_quantity > comb_limit:
        raise ValueError(f" merge_quantity is {merge_quantity}: {merge_quantity} is bigger than comb_limit {comb_limit}")

    # Take summed values over one axis
    sum_cmat = label_aff_mat.sum(0)

    # (merge_quantity + 1) will become 1 embedding vector after merging
    idx_aff_sum = torch.argsort(sum_cmat, descending=True)[:(merge_quantity+1)]
    return idx_aff_sum


@torch.jit.script
def run_reducer(
    pre_embs: torch.Tensor, 
    target_spk_idx: int, 
    merge_quantity: int, 
    pre_clus_labels: torch.Tensor, 
    ):
    """
    Reduce the number of embedding vectors by merging the closest embedding vectors.
    - This merging algorithm is based on the assumption that the closest embeddings are the most redundant
    embedding vectors.
    - The closest embedding vectors are chosen by selecting the highest top-N sum of each column in a given
      affinity matrix.
    - If merge_quantity is N, we choose (N+1) vectors into 1 embedding vector. Thus, we reduce N embeddings
      in the original embedding vector set.

    Example:
        >>> merge_quantity = 1 # We merge 1+1 = 2 embedding vectors
        >>> affinity_mat = [[1.0, 0.2, 0.8],
                            [0.2, 1.0, 0.4],
                            [0.8, 0.4, 1.0]]
        >>> affinity_mat.sum(0) 
        [2.0, 1.6, 2.2]

        The first and the third embedding vectors are merged into one embedding vector.
        >>> index_mapping # (bypassed indices, merged indices)
        ([1], [0, 2]) 

    Args:
        pre_embs (Tensor):
            Potential Embedding vectors to be merged
        affinity_mat (Tensor):
            The affinity matrix of the `pre_embs`
        target_spk_idx (int):
            The targeted speaker index for merging
        merge_quantity (int):
            The count of embeddings to be reduced
        pre_clus_labels (list)
            The original cluster (speaker) index

    Returns:
        result_emb (Tensor):
            Set of merged embedding vectors
        merged_clus_labels (list):
            Cluster (speaker) labels for the merged embedding vectors
    """
    if pre_embs.shape[0] != pre_clus_labels.shape[0]:
        raise ValueError("Dimension mismatch between `pre_embs` and `pre_clus_labels`.")

    target_emb_index = torch.where(pre_clus_labels == target_spk_idx)[0]
    org_size = target_emb_index.shape[0]
    if merge_quantity > 0:
        if merge_quantity > (target_emb_index.shape[0]-1):
            raise ValueError(
                f"merge_quantity {merge_quantity} is larger than the half of targeted speaker's labels {target_emb_index.shape[0]-1}"
            )
        affinity_mat = getCosAffinityMatrix(pre_embs)
        # Get the lower triangle of the affinity_mat array
        label_aff_mat = affinity_mat[:, target_emb_index][target_emb_index, :]
        if label_aff_mat.shape[0] != target_emb_index.shape[0]:
            raise ValueError(
                "Dimension mismatch between targeted speaker affinity `label_aff_mat` and targeted speaker index `target_emb_index`."
            )
        # Get the indices of the closest embedding vectors
        selected_inds = get_closest_embeddings(label_aff_mat, target_emb_index, merge_quantity)
        spk_cluster_labels, selected_embs = pre_clus_labels[target_emb_index], pre_embs[target_emb_index]
        
        # Merge the embeddings targeted by the 2-dim indices `index_2d`
        merged_embs, merged_clus_labels, index_mapping = merge_vectors(selected_inds, selected_embs, spk_cluster_labels)
        target_emb_index = torch.where(merged_clus_labels == target_spk_idx)[0]
        if (org_size - merge_quantity) != merged_embs.shape[0]:
            raise ValueError("Reducer output is not matched to the target quantity")

    else:
        merged_embs = pre_embs[target_emb_index]
        merged_clus_labels = pre_clus_labels[target_emb_index]
        index_mapping = (torch.arange(merged_embs.shape[0]), torch.arange(0))
    return merged_embs, merged_clus_labels, index_mapping


@torch.jit.script
def get_first_arg_index(mat: torch.Tensor, label: int) -> int:
    """
    Get the index of the first element are specified by `index` variable.

    Args:
        mat (Tensor):
            Source matrix filled with indices
        label (int):
            Label which we want to find the first occuring index

    Returns:
        (int) The first index of the given label
    """
    return int(torch.where(mat == label)[0][0])


@torch.jit.script
class SpectralClustering:
    """
    Perform spectral clustering by calculating spectral embeddings then run k-means clustering
    algorithm on the spectral embeddings.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        random_state: int = 0,
        n_random_trials: int = 1,
        cuda: bool = False,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize the variables needed for spectral clustering and k-means++.

        Args:
            n_clusters (int):
                Number of the estimated (or oracle) number of speakers
            random_state (int):
                Random seed that determines a random state of k-means initialization.
            n_random_trials (int):
                Number of trials with different random seeds for k-means initialization.
                k-means++ algorithm is executed for multiple times then the final result
                is obtained by taking a majority vote.
            cuda (bool):
                if cuda=True, spectral clustering is done on GPU.
            device (torch.device):
                Torch device variable
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_random_trials = max(n_random_trials, 1)
        self.cuda = cuda
        self.device = device

    def forward(self, X) -> torch.Tensor:
        """
        Call self.clusterSpectralEmbeddings() function to predict cluster labels.

        Args:
            X (Tensor):
                Affinity matrix input

        Returns:
            labels (Tensor):
                clustering label output
        """
        if X.shape[0] != X.shape[1]:
            raise ValueError("The affinity matrix is not a square matrix.")
        labels = self.clusterSpectralEmbeddings(X, cuda=self.cuda, device=self.device)
        return labels

    def clusterSpectralEmbeddings(
        self, affinity: torch.Tensor, cuda: bool = False, device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Perform k-means clustering on spectral embeddings. To alleviate the effect of randomness,
        k-means clustering is performed for (self.n_random_trials) times then the final labels are obtained
        by taking a majority vote. If speed is the major concern, self.n_random_trials should be set to 1.
        n_random_trials=30 is recommended to see an improved result.

        Args:
            affinity (Tensor):
                Affinity matrix input
            cuda (torch.bool):
                Use cuda for spectral clustering if cuda=True
            device (torch.device):
                Torch device variable

        Returns:
            labels (Tensor):
                clustering label output

        """
        spectral_emb = self.getSpectralEmbeddings(affinity, n_spks=self.n_clusters, cuda=cuda)
        labels_set = []
        for random_state_seed in range(self.random_state, self.random_state + self.n_random_trials):
            _labels = kmeans_torch(
                X=spectral_emb, num_clusters=self.n_clusters, random_state=random_state_seed, device=device
            )
            labels_set.append(_labels)
        stacked_labels = torch.stack(labels_set)
        label_index = torch.mode(torch.mode(stacked_labels, 0)[1])[0]
        labels = stacked_labels[label_index]
        return labels

    def getSpectralEmbeddings(self, affinity_mat: torch.Tensor, n_spks: int = 8, cuda: bool = False) -> torch.Tensor:
        """
        Calculate eigenvalues and eigenvectors to extract spectral embeddings.

        Args:
            affinity (Tensor):
                Affinity matrix input
            cuda (torch.bool):
                Use cuda for spectral clustering if cuda=True
            device (torch.device):
                Torch device variable

        Returns:
            labels (Tensor):
                clustering label output
        """
        laplacian = getLaplacian(affinity_mat)
        lambdas_, diffusion_map_ = eigDecompose(laplacian, cuda=cuda)
        diffusion_map = diffusion_map_[:, :n_spks]
        inv_idx = torch.arange(diffusion_map.size(1) - 1, -1, -1).long()
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
    NME-analysis. Fixing p_value brings about significantly faster clustering
    speed, but the performance is limited to the development set.

    References:
        Tae Jin Park et al., Auto-Tuning Spectral Clustering for Speaker Diarization
        Using Normalized Maximum Eigengap, IEEE Signal Processing Letters 27 (2019),
        https://arxiv.org/abs/2003.02405

    Args:
        Please refer to def __init__().

    Methods:
        NMEanalysis():
            Performs NME-analysis to estimate p_value and the number of speakers
        subsampleAffinityMat(nme_mat_size):
            Subsamples the number of speakers to reduce the computational load
        getPvalueList():
            Generates a list containing p-values that need to be examined.
        getEigRatio(p_neighbors):
            Calculates g_p, which is a ratio between p_neighbors and the maximum eigengap
        getLamdaGaplist(lambdas):
            Calculates lambda gap values from an array contains lambda values
        estimateNumofSpeakers(affinity_mat):
            Estimates the number of speakers using lambda gap list
    """

    def __init__(
        self,
        mat: torch.Tensor,
        max_num_speakers: int = 10,
        max_rp_threshold: float = 0.15,
        sparse_search: bool = True,
        sparse_search_volume: int = 30,
        nme_mat_size: int = 512,
        use_subsampling_for_nme: bool = True,
        fixed_thres: float = -1.0,
        maj_vote_spk_count: bool = False,
        parallelism: bool = True,
        cuda: bool = False,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            mat (Tensor):
                Cosine similarity matrix calculated from the provided speaker embeddings.
            max_num_speakers (int):
                Maximum number of speakers for estimating number of speakers.
                Shows stable performance under 20.
            max_rp_threshold (float):
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.25.
            sparse_search (bool):
                To increase the speed of parameter estimation, sparse_search=True
                limits the number of p_values we search.
            sparse_search_volume (int):
                Number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                However, a value lower than 20 might cause a poor parameter estimation.
            nme_mat_size (int):
                Targeted size of matrix for NME analysis.
            use_subsampling_for_nme (bool):
                Use subsampling to reduce the calculational complexity.
                Default is True.
            fixed_thres (float or None):
                A fixed threshold which can be used instead of estimating the
                threshold with NME analysis. If fixed_thres is float,
                it skips the NME analysis part.
            maj_vote_spk_count (bool):
                If True, take a majority vote on all p-values in the given range to estimate the number of speakers.
                The majority voting may contribute to surpress overcounting of the speakers and improve speaker
                counting accuracy.
            parallelism (bool):
                If True, turn on parallelism based on torch.jit.script library.
            cuda (bool):
                Use cuda for Eigen decomposition if cuda=True.
            device (torch.device):
                Torch device variable

        """
        self.max_num_speakers: int = max_num_speakers
        self.max_rp_threshold: float = max_rp_threshold
        self.use_subsampling_for_nme: bool = use_subsampling_for_nme
        self.nme_mat_size: int = nme_mat_size
        self.sparse_search: bool = sparse_search
        self.sparse_search_volume: int = sparse_search_volume
        self.min_p_value = torch.tensor(2)
        self.fixed_thres: float = fixed_thres
        self.eps = 1e-10
        self.max_N = torch.tensor(0)
        self.mat: torch.Tensor = mat
        self.p_value_list: torch.Tensor = self.min_p_value.unsqueeze(0)
        self.cuda: bool = cuda
        self.device: torch.device = device
        self.maj_vote_spk_count: bool = maj_vote_spk_count
        self.parallelism: bool = parallelism

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Subsample the input matrix to reduce the computational load.

        Returns:
            est_num_of_spk (Tensor):
                Estimated number of speakers from NMESC approach
            p_hat_value (Tensor):
                Estimated p-value (determines how many neighboring values to be selected)
        """
        if self.use_subsampling_for_nme:
            subsample_ratio = self.subsampleAffinityMat(self.nme_mat_size)
        else:
            subsample_ratio = torch.tensor(1)

        # Scans p_values and find a p_value that generates the smallest g_p value.
        results: List[torch.Tensor] = []
        est_spk_n_dict: Dict[int, torch.Tensor] = {}
        self.p_value_list = self.getPvalueList()
        p_volume = self.p_value_list.shape[0]
        eig_ratio_list = torch.zeros(p_volume,)
        est_num_of_spk_list = torch.zeros(p_volume,)

        if self.parallelism:
            futures: List[torch.jit.Future[torch.Tensor]] = []
            for p_idx, p_value in enumerate(self.p_value_list):
                futures.append(torch.jit.fork(self.getEigRatio, p_value))
            for future in futures:
                results.append(torch.jit.wait(future))

        else:
            for p_idx, p_value in enumerate(self.p_value_list):
                results.append(self.getEigRatio(p_value))

        # Retrieve the eigen analysis results
        for p_idx, p_value in enumerate(self.p_value_list):
            output = results[p_idx]
            g_p, est_num_of_spk = output[0], output[1].int()
            eig_ratio_list[p_idx] = g_p
            est_spk_n_dict[p_value.item()] = est_num_of_spk
            est_num_of_spk_list[p_idx] = est_num_of_spk

        index_nn = torch.argmin(eig_ratio_list)
        rp_p_value = self.p_value_list[index_nn]
        affinity_mat = getAffinityGraphMat(self.mat, rp_p_value)

        # Checks whether the affinity graph is fully connected.
        # If not, it adds a minimum number of connections to make it fully connected.
        if not isGraphFullyConnected(affinity_mat, device=self.device):
            affinity_mat, rp_p_value = getMinimumConnection(
                self.mat, self.max_N, self.p_value_list, device=self.device
            )

        p_hat_value = (subsample_ratio * rp_p_value).type(torch.int)
        if self.maj_vote_spk_count:
            est_num_of_spk = torch.mode(torch.tensor(est_num_of_spk_list))[0]
        else:
            est_num_of_spk = est_spk_n_dict[rp_p_value.item()]
        return est_num_of_spk, p_hat_value

    def subsampleAffinityMat(self, nme_mat_size: int) -> torch.Tensor:
        """
        Perform subsampling of affinity matrix.
        This subsampling is for calculational complexity, not for performance.
        The smaller nme_mat_size is,
            - the bigger the chance of missing a speaker.
            - the faster p-value estimation speed (based on eigen decomposition).

        The recommended nme_mat_size is 250~750.
        However, if there are speakers who speak for very short period of time in the recording,
        this subsampling might make the system miss underrepresented speakers.
        Use this variable with caution.

        Args:
            nme_mat_size (int):
                The targeted matrix size

        Returns:
            subsample_ratio (float):
                The ratio between nme_mat_size and the original matrix size
        """
        subsample_ratio = torch.max(torch.tensor(1), torch.tensor(self.mat.shape[0] / nme_mat_size)).type(torch.int)
        self.mat = self.mat[:: subsample_ratio.item(), :: subsample_ratio.item()]
        return subsample_ratio

    def getEigRatio(self, p_neighbors: int) -> torch.Tensor:
        """
        For a given p_neighbors value, calculate g_p, which is a ratio between p_neighbors and the
        maximum eigengap values.
        References:
            Tae Jin Park et al., Auto-Tuning Spectral Clustering for Speaker Diarization Using
            Normalized Maximum Eigengap, IEEE Signal Processing Letters 27 (2019),
            https://arxiv.org/abs/2003.02405

        Args:
            p_neighbors (int):
                Determines how many binary graph connections we want to keep for each row.

        Returns:
            est_num_of_spk (int):
                Estimated number of speakers
            g_p (float):
                The ratio between p_neighbors value and the maximum eigen gap value.
        """
        affinity_mat = getAffinityGraphMat(self.mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = estimateNumofSpeakers(
            affinity_mat, self.max_num_speakers, self.cuda
        )
        arg_sorted_idx = torch.argsort(lambda_gap_list[: self.max_num_speakers], descending=True)
        max_key = arg_sorted_idx[0]
        max_eig_gap = lambda_gap_list[max_key] / (torch.max(lambdas).item() + self.eps)
        g_p = (p_neighbors / self.mat.shape[0]) / (max_eig_gap + self.eps)
        return torch.stack([g_p, est_num_of_spk])

    def getPvalueList(self) -> torch.Tensor:
        """
        Generates a p-value (p_neighbour) list for searching. p_value_list must include 2 (min_p_value)
        since at least one neighboring segment should be selected other than itself.

        If fixed_thres value is specified, then only one p-value is specified.
        If fixed_thres is not provided, multiple p-values are searched.
            If sparse_search is True:
                - Limit the number of p-values to be searched to sparse_search_volume.
                - N should be at least 2 to include a number greater than 1.
            If sparse_search is False:
                - Scan all the p_values from 1 to max_N
                - If sparse_search is False, NMESC analysis could take more time compared to sparse_search = True.

        Returns:
            p_value_list (Tensor):
                Tensor containing the p_values to be searched.
        """
        if self.fixed_thres is not None and self.fixed_thres > 0.0:
            self.max_N = torch.max(
                torch.floor(torch.tensor(self.mat.shape[0] * self.fixed_thres)).type(torch.int), self.min_p_value
            )
            p_value_list = torch.tensor(self.max_N).unsqueeze(0).int()
        else:
            self.max_N = torch.max(
                torch.floor(torch.tensor(self.mat.shape[0] * self.max_rp_threshold)).type(torch.int), self.min_p_value
            )
            if self.sparse_search:
                search_volume = torch.min(self.max_N, torch.tensor(self.sparse_search_volume).type(torch.int))
                # search at least two values
                N = torch.max(search_volume, torch.tensor(2))
                # avoid repeating values by limiting the step size
                steps = min(self.max_N, N)
                p_value_list = torch.linspace(start=1, end=self.max_N, steps=steps).type(torch.int)
            else:
                p_value_list = torch.arange(1, self.max_N + 1)
        if p_value_list.shape[0] == 0:
            raise ValueError("p_value_list should not be empty.")
        return p_value_list


class SpeakerClustering(torch.nn.Module):
    def __init__(
        self,
        min_samples_for_nmesc: int = 6,
        nme_mat_size: int = 300,
        sparse_search: bool = True,
        maj_vote_spk_count: bool = False,
        parallelism: bool = True,
        cuda: bool = False,
    ):
        """
        Clustering method for speaker diarization based on cosine similarity.
        NME-SC part is converted to torch.tensor based operations in NeMo 1.9.

        Args:
            min_samples_for_nmesc (int):
                The minimum number of samples required for NME clustering. This avoids
                zero p_neighbour_lists. If the input has fewer segments than min_samples,
                it is directed to the enhanced speaker counting mode.
            sparse_search (bool):
                Toggle sparse search mode. If True, limit the size of p_value_list to sparse_search_volume.
            maj_vote_spk_count (bool):
                If True, take a majority vote on all p-values in the given range to estimate the number of speakers.
                The majority voting may contribute to surpress overcounting of the speakers and improve speaker
                counting accuracy.
            parallelism (bool):
                Use dynamic parallelism feature in torch.jit compiler to accelerate the p-value search.
            cuda (bool):
                Boolean variable for toggling cuda availability.
        """
        super().__init__()
        self.min_samples_for_nmesc: int = min_samples_for_nmesc
        self.nme_mat_size: int = nme_mat_size
        self.sparse_search: bool = sparse_search
        self.parallelism: bool = parallelism
        self.cuda: bool = cuda
        self.maj_vote_spk_count: bool = maj_vote_spk_count
        self.embeddings_in_scales: List[torch.Tensor] = [torch.Tensor(0)]
        self.timestamps_in_scales: List[torch.Tensor] = [torch.Tensor(0)]
        self.device = torch.device("cuda") if self.cuda else torch.device("cpu")

    def forward(self, param_dict: Dict[str, torch.Tensor]) -> torch.LongTensor:
        """
        A function wrapper designed for inference in exported script format.

        Note:
            Dict is used to allow easy inference of the exported jit model in Triton server using easy to understand
            naming convention.
            See https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#special-conventions-for-pytorch-backend


        Args:
            param_dict (dict):
                    Dictionary containing the arguments for speaker clustering.
                    See `forward_infer` function for the argument information.

            Returns:
                Y (LongTensor):
                    Speaker labels for the segments in the given input embeddings.
        """
        embeddings_in_scales = param_dict['embeddings']
        timestamps_in_scales = param_dict['timestamps']
        multiscale_segment_counts = param_dict['multiscale_segment_counts']
        multiscale_weights = param_dict['multiscale_weights']

        oracle_num_speakers = int(param_dict['oracle_num_speakers'].item())
        max_num_speakers = int(param_dict['max_num_speakers'].item())
        enhanced_count_thres = int(param_dict['enhanced_count_thres'].item())
        sparse_search_volume = int(param_dict['sparse_search_volume'].item())
        max_rp_threshold = float(param_dict['max_rp_threshold'].item())
        fixed_thres = float(param_dict['fixed_thres'].item())

        return self.forward_infer(
            embeddings_in_scales=embeddings_in_scales,
            timestamps_in_scales=timestamps_in_scales,
            multiscale_segment_counts=multiscale_segment_counts,
            multiscale_weights=multiscale_weights,
            oracle_num_speakers=oracle_num_speakers,
            max_rp_threshold=max_rp_threshold,
            max_num_speakers=max_num_speakers,
            enhanced_count_thres=enhanced_count_thres,
            sparse_search_volume=sparse_search_volume,
            fixed_thres=fixed_thres,
        )

    def forward_infer(
        self,
        embeddings_in_scales: torch.Tensor,
        timestamps_in_scales: torch.Tensor,
        multiscale_segment_counts: torch.LongTensor,
        multiscale_weights: torch.Tensor,
        oracle_num_speakers: int = -1,
        max_rp_threshold: float = 0.15,
        max_num_speakers: int = 8,
        enhanced_count_thres: int = 80,
        sparse_search_volume: int = 30,
        fixed_thres: float = -1.0,
    ) -> torch.LongTensor:
        """
        Calculate affinity matrix using timestamps and speaker embeddings, run NME analysis to estimate the best
        p-value and perform spectral clustering based on the estimated p-value and the calculated affinity matrix.

        Caution:
            For the sake of compatibility with libtorch, python boolean `False` is replaced with `torch.LongTensor(-1)`.

        Args:
            Dict containing following keys associated with tensors.
            embeddings (Tensor):
                Concatenated Torch tensor containing embeddings in multiple scales
                This tensor has dimensions of (Number of base segments) x (Embedding Dimension)
            timestamps (Tensor):
                Concatenated Torch tensor containing timestamps in multiple scales.
                This tensor has dimensions of (Total number of segments all scales) x 2
                Example:
                    >>> timestamps_in_scales = \
                    >>> torch.tensor([0.4, 1.4], [0.9, 1.9], [1.4, 2.4], ... [121.2, 122.2]])

            multiscale_segment_counts (LongTensor):
                Concatenated Torch tensor containing number of segments per each scale
                This tensor has dimensions of (Number of scales)
                Example:
                    >>> multiscale_segment_counts = torch.LongTensor([31, 52, 84, 105, 120])

            multiscale_weights (Tensor):
                Multi-scale weights that are used when affinity scores are merged.
                Example:
                    >>> multiscale_weights = torch.tensor([1.4, 1.3, 1.2, 1.1, 1.0])

            oracle_num_speakers (int):
                The number of speakers in a session from the reference transcript
            max_num_speakers (int):
                The upper bound for the number of speakers in each session
            max_rp_threshold (float):
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.15.
            enhanced_count_thres (int):
                For the short audio recordings, clustering algorithm cannot
                accumulate enough amount of speaker profile for each cluster.
                Thus, function `getEnhancedSpeakerCount` employs anchor embeddings
                (dummy representations) to mitigate the effect of cluster sparsity.
                enhanced_count_thres = 80 is recommended.
            sparse_search_volume (int):
                Number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                Lower than 20 might cause a poor parameter estimation.
            fixed_thres (float):
                If fixed_thres value is provided, NME-analysis process will be skipped.
                This value should be optimized on a development set to obtain a quality result.
                Default is None and performs NME-analysis to estimate the threshold.

        Returns:
            Y (LongTensor):
                Speaker labels for the segments in the given input embeddings.
        """
        self.embeddings_in_scales, self.timestamps_in_scales = split_input_data(
            embeddings_in_scales, timestamps_in_scales, multiscale_segment_counts
        )

        emb, _ = getTempInterpolMultiScaleCosAffinityMatrix(multiscale_weights, 
                                                            self.embeddings_in_scales, 
                                                            self.timestamps_in_scales, 
                                                            self.device)

        # Cases for extreamly short sessions
        if emb.shape[0] == 1:
            return torch.zeros((1,), dtype=torch.int64)
        elif emb.shape[0] <= max(enhanced_count_thres, self.min_samples_for_nmesc) and oracle_num_speakers < 0:
            est_num_of_spk_enhanced = getEnhancedSpeakerCount(emb=emb, cuda=self.cuda)
        else:
            est_num_of_spk_enhanced = torch.tensor(-1)

        if oracle_num_speakers > 0:
            max_num_speakers = oracle_num_speakers

        mat = getMultiScaleCosAffinityMatrix(
            multiscale_weights, self.embeddings_in_scales, self.timestamps_in_scales, self.device
        )

        nmesc = NMESC(
            mat,
            max_num_speakers=max_num_speakers,
            max_rp_threshold=max_rp_threshold,
            sparse_search=self.sparse_search,
            sparse_search_volume=sparse_search_volume,
            fixed_thres=fixed_thres,
            nme_mat_size=self.nme_mat_size,
            maj_vote_spk_count=self.maj_vote_spk_count,
            parallelism=self.parallelism,
            cuda=self.cuda,
            device=self.device,
        )

        # If there are less than `min_samples_for_nmesc` segments, est_num_of_spk is 1.
        if mat.shape[0] > self.min_samples_for_nmesc:
            est_num_of_spk, p_hat_value = nmesc.forward()
            affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        else:
            est_num_of_spk = torch.tensor(1)
            affinity_mat = mat

        # n_clusters is number of speakers estimated from spectral clustering.
        if oracle_num_speakers > 0:
            n_clusters = int(oracle_num_speakers)
        elif est_num_of_spk_enhanced > 0:
            n_clusters = int(est_num_of_spk_enhanced.item())
        else:
            n_clusters = int(est_num_of_spk.item())

        spectral_model = SpectralClustering(n_clusters=n_clusters, cuda=self.cuda, device=self.device)
        Y = spectral_model.forward(affinity_mat)
        return Y


class OnlineSpeakerClustering:
    """
    Online clustering method for speaker diarization based on cosine similarity.

    Regular Clustering Attributes:

        max_num_speakers (int):
            The upper bound for the number of speakers in each session
        max_rp_threshold (float):
            Limits the range of parameter search.
            Clustering performance can vary depending on this range.
            Default is 0.15.
        enhanced_count_thres (int):
            For the short audio recordings, clustering algorithm cannot
            accumulate enough amount of speaker profile for each cluster.
            Thus, function `getEnhancedSpeakerCount` employs anchor embeddings
            (dummy representations) to mitigate the effect of cluster sparsity.
            enhanced_count_thres = 80 is recommended.
        sparse_search_volume (int):
            Number of p_values we search during NME analysis.
            Default is 30. The lower the value, the faster NME-analysis becomes.
            Lower than 20 might cause a poor parameter estimation.
        fixed_thres (float):
            If fixed_thres value is provided, NME-analysis process will be skipped.
            This value should be optimized on a development set to obtain a quality result.
            Default is None and performs NME-analysis to estimate the threshold.
        min_samples_for_nmesc (int):
            The minimum number of samples required for NME clustering. This avoids
            zero p_neighbour_lists. If the input has fewer segments than min_samples,
            it is directed to the enhanced speaker counting mode.
        sparse_search (bool):
            Toggle sparse search mode. If True, limit the size of p_value_list to sparse_search_volume.
        cuda (bool):
            Use cuda for Eigen decomposition if cuda=True.
        device (torch.device):
            Torch device variable

    Online Processing Attributes:

        history_buffer_size (int):
            - This is a buffer where diarization history is saved in the form of averaged speaker embedding vector.
            - The values in [50, 200] range is recommended while the system requires bigger buffer size for
              sessions with larger number of speakers.
        current_buffer_size (int):
            - This is a buffer which process the most recent speaker embedding vector inputs.
              current-buffer is first-in-first-out (FIFO) queue where the embeddings accepted earlier
              get to merged and saved to history buffer.
            - In general, [50, 200] range is recommended and the performance can be sensitive on this buffer size.
        min_spk_counting_buffer_size (int):
            Integer number for speaker counting buffer. Number of speakers are estimated through a small buffer
            and the number is obtained by taking majority vote.
        min_frame_per_spk (int):
            Below this number, the system considers the whole input segments as a single speaker.
        p_update_freq (int):
            Frequency (interval) of updating p_value for NMESC algorithm.
        p_value_skip_frame_thres (int):
            After `frame_index` passes this number, `p_value` estimation is skipped for inference speed
        p_value_queue_size (int):
            `p_value` buffer for major voting
        use_temporal_label_major_vote (bool):
            Boolean that determines whether to use temporal majorvoting for the final speaker labels
        temporal_label_major_vote_buffer_size (int):
            Buffer size for major-voting the
    """
    def __init__(
        self,
        max_num_speakers: int,
        max_rp_threshold: float = 0.15,
        enhanced_count_thres: float = 80,
        fixed_thres: float = -1.0,
        sparse_search_volume: int = 15,
        history_buffer_size: int = 150,
        current_buffer_size: int = 150,
        min_spk_counting_buffer_size = 7,
        min_frame_per_spk: int = 20,
        p_update_freq: int = 5,
        p_value_skip_frame_thres: int = 50,
        p_value_queue_size: int = 3,
        use_temporal_label_major_vote: bool = False,
        temporal_label_major_vote_buffer_size: int = 11,
        cuda=False,
        device: torch.device = torch.device("cpu"),
    ):
        self.max_num_speakers = max_num_speakers
        self.max_rp_threshold = max_rp_threshold
        self.enhanced_count_thres = enhanced_count_thres
        self.sparse_search_volume = sparse_search_volume
        self.fixed_thres = fixed_thres
        self.history_n = history_buffer_size
        self.current_n = current_buffer_size
        self.min_spk_counting_buffer_size = min_spk_counting_buffer_size
        self.min_frame_per_spk = min_frame_per_spk
        self.p_update_freq = p_update_freq
        self.p_value_skip_frame_thres = p_value_skip_frame_thres
        self.p_value_queue_size = p_value_queue_size
        self.use_temporal_label_major_vote = use_temporal_label_major_vote
        self.temporal_label_major_vote_buffer_size = temporal_label_major_vote_buffer_size
        self.num_spk_stat = []
        self.p_value_hist = []

        self.cuda = cuda
        self.device = device

        self._init_memory_buffer_variables()
        self._init_memory_embeddings()

    def _init_memory_buffer_variables(self):
        """
        Initialize memory buffer related variables.

        Attributes:
            max_embed_count (int):
                The maximum number of segments the streaming system has ever seen
            memory_margin (int):
                The margin that is added to keep the segmentation data in the streaming system
            _minimum_segments_per_buffer (int):
                Maximum number of embedding vectors kept in history buffer per speaker.
                Example:
                    history_buffer_size (history_n) = 100
                    max_num_speakers = 4
                    _minimum_segments_per_buffer = 25
            history_buffer_seg_end (int):
                Index that indicates the boundary between history embedding sets and current processing buffer
                when history embedding vectors and current input embedding vectors are concatenated into a
                single matrix.
        """
        self.max_embed_count = 0
        self.memory_margin = 0
        self._minimum_segments_per_buffer = int(self.history_n / self.max_num_speakers)
        self.history_buffer_seg_end = 0

    def _init_memory_embeddings(self):
        """
        Initialize history buffer related variables.

        Attributes:
            isOnline (bool):
                If self.isOnline is False:
                    FIFO queue does not push out any speaker embedding vector
                If self.isOnline is True:
                    FIFO queue starts push out speaker embedding vectors and saving them into
                    history buffer.
            history_embedding_buffer_emb (Tensor)
                Tensor containing speaker embedding vectors for saving the history of the previous
                speaker profile in the given audio session
            history_embedding_buffer_label (Tensor)
                Speaker label (cluster label) for embedding vectors saved in the history buffer
            Y_fullhist (Tensor)
                Tensor containing the speaker label hypothesis from start to current frame
        """
        self.isOnline = False
        self.history_embedding_buffer_emb = torch.tensor([])
        self.history_embedding_buffer_label = torch.tensor([])
        self.Y_fullhist = torch.tensor([])

    def onlineNMEanalysis(self, nmesc: NMESC, frame_index: int) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        To save the running time, the p-value is only estimated in the beginning of the session.
        After switching to online mode, the system uses the most common estimated p-value.
        Estimating p-value requires a plenty of computational resource. The less frequent estimation of
        p-value can speed up the clustering algorithm by a huge margin.

        Args:
            nmesc: (NMESC)
                nmesc instance.
            isOnline: (bool)
                Indicates whether the system is running on online mode or not.

        Returns:
            est_num_of_spk: (int)
                The estimated number of speakers.
            p_hat_value: (int)
                The estimated p-value from NMESC method.
        """
        if len(self.p_value_hist) == 0 or (
            frame_index < self.p_value_skip_frame_thres and frame_index % self.p_update_freq == 0
        ):
            est_num_of_spk, p_hat_value = nmesc.forward()
            self.p_value_hist.append(p_hat_value)
            if len(self.p_value_hist) > self.p_value_queue_size:
                self.p_value_hist.pop(0)
        p_hat_value = max(self.p_value_hist, key=self.p_value_hist.count)
        g_p, est_num_of_spk = nmesc.getEigRatio(p_hat_value)
        return est_num_of_spk, p_hat_value

    def speaker_counter_buffer(self, est_num_of_spk: int) -> int:
        """
        Use a queue to avoid unstable speaker counting results.

        Args:
            est_num_of_spk (int):
                Estimated number of speakers
        """
        if type(est_num_of_spk.item()) != int:
            est_num_of_spk = int(est_num_of_spk.item())

        self.num_spk_stat.append(est_num_of_spk)
        if len(self.num_spk_stat) > self.min_spk_counting_buffer_size:
            self.num_spk_stat.pop(0)
        num_spks_bincount = torch.bincount(torch.tensor(self.num_spk_stat))
        est_num_of_spk = torch.argmax(num_spks_bincount)
        return est_num_of_spk

    def limit_frames_per_speaker(self, frame_index: int, est_num_of_spk: int) -> int:
        """
        Limit the estimated number of speakers in proportion to the number of speakers.

        Args:
            est_num_of_spk (int): Estimated number of speakers
        Returns:
            (int) Estimated number of speakers capped by `self.min_frame_per_spk`
        """
        return min(est_num_of_spk, int(1 + frame_index // self.min_frame_per_spk))

    def online_spk_num_estimation(self, mat_in, nmesc, frame_index):
        """
        Online version of speaker estimation involves speaker counting buffer and application of per-speaker
        frame count limit.

        Args:
            mat_in (Tensor):
                Raw affinity matrix containing similarity values of each pair of segments
            nmesc (NMESC):
                NMESC class instance
            frame_index (int)
                Unique frame index of online processing pipeline

        Returns:
            est_num_of_spk (int):
                Estimated number of speakers
            nmesc (NMESC):
                NMESC class instance
            frame_index (int):
                Unique frame index of online processing pipeline
        """
        est_num_of_spk, p_hat_value = self.onlineNMEanalysis(nmesc, frame_index)
        affinity_mat = getAffinityGraphMat(mat_in, p_hat_value)
        est_num_of_spk = self.speaker_counter_buffer(est_num_of_spk)
        est_num_of_spk = self.limit_frames_per_speaker(frame_index, est_num_of_spk)
        return est_num_of_spk, affinity_mat

    def prepare_embedding_update(self, emb_in: torch.Tensor, base_segment_indexes: List[int]):
        """
        This function performs the following tasks:
            1. Decide whether to extract more embeddings or not (by setting `update_speaker_register`)
        (If we need update):
            2. Calculate how many embeddings should be updated (set `new_emb_n` variable)
            3. Update history embedding vectors and save it to `pre_embs`.

        We only save the index and clustering label of each embedding.

        - Case-1: The very first step
            This else statement is for the very first diarization loop.
            This is the very first reduction frame.

        - Case-2: Number of embedding vectors is increased, therefore we need to update.
            Since there are new embeddings, we push the same amount (new_emb_n)
            of old embeddings to the history buffer.
            We should also update self.history_buffer_seg_end which is a pointer.
                update to history emb: emb_in[emb_idx_stt:emb_idx_end]
                update to history label: self.Y_fullhist[label_stt:_end]

        - Case-3: Number of embedding vectors is decreased
            If the number of embeddings is decreased compared to the last trial,
            then skip embedding merging.

        Args:
            emb_in (Tensor):
                Tensor containing embedding vectors
                Dimensions: (number of embedding vectors) x (embedding dimension)
            base_segment_indexes (list):
                List containing unique segment (embedding vector) index

        Returns:
            update_speaker_register (bool):
                Boolean indicates whether to update speaker embedding vectors.
            new_emb_n (int):
                The amount of embedding vectors that are exceeding FIFO queue size
            pre_embs (Tensor):
                Embedding vector matrix (# of embs x emb dim) before merging
        """
        _segment_indexes_mat = torch.tensor(base_segment_indexes)
        self.total_segments_processed_count = int(_segment_indexes_mat[-1] + 1)
        hist_curr_boundary = int(self.total_segments_processed_count - self.current_n)
        new_emb_n, pre_embs = None, None
        update_speaker_register = True

        # Case-1: The very first step
        if len(self.history_embedding_buffer_emb) == 0:
            new_emb_n = self.total_segments_processed_count - (self.current_n + self.history_n)
            hist_curr_boundary_emb_idx = get_first_arg_index(_segment_indexes_mat, hist_curr_boundary)
            pre_embs = emb_in[:hist_curr_boundary_emb_idx]
            self.pre_clus_labels = self.Y_fullhist[:hist_curr_boundary]

        # Case-2: Number of embedding vectors is increased, need to update history and its label
        elif self.total_segments_processed_count > self.max_embed_count:

            # Calculate the number of new embedding vectors
            label_stt, label_end = self.history_buffer_seg_end, hist_curr_boundary
            new_emb_n = label_end - label_stt
            assert new_emb_n > 0, "new_emb_n should be a positve integer number."

            # Add embedding vectors to `pre_embs` so that we can merge it with reducer function.
            emb_idx_stt = int(get_first_arg_index(_segment_indexes_mat, label_stt))
            emb_idx_end = int(get_first_arg_index(_segment_indexes_mat, label_end))
            pre_embs = torch.vstack((self.history_embedding_buffer_emb, emb_in[emb_idx_stt:emb_idx_end]))
            self.pre_clus_labels = torch.hstack(
                (self.history_embedding_buffer_label, self.Y_fullhist[label_stt:label_end])
            )

        # Case-3: Number of embedding vectors is decreased
        # There will be no embedding update, so new_emb_n, pre_embs should be None
        else:
            update_speaker_register = False

        # Update the history buffer index
        self.history_buffer_seg_end = hist_curr_boundary
        if new_emb_n is not None and new_emb_n < 0:
            raise ValueError(f"new_emb_n should not be negative but got new_emb_n: {new_emb_n}")
        return update_speaker_register, new_emb_n, pre_embs

    def make_constant_length_emb(self, emb_in, base_segment_indexes):
        """
        This function deals with edge cases when the number of segments decreases and the number of embedding falls
        short for the labels.

        - ASR decoder occasionally returns less number of words compared to the previous frame.
        - In this case, we obtain fewer embedding vectors for the short period of time. To match the pre-defined
          length, the last embedding vector is repeated to fill the voidness.
        - The repeated embedding will be soon replaced by the actual embeddings once the system takes new frames.

        Args:
            emb_in (Tensor):
                If self.isOnline is False:
                    `emb` contains only current speaker embedding inputs, which is FIFO queue
                If self.isOnline is True:
                    `emb` contains history buffer and FIFO queue
            base_segment_indexes (Tensor):
                Tensor containing unique segment (embedding vector) index

        Returns:
            emb_curr (Tensor):
                Length preserved speaker embedding vectors
        """
        segment_indexes_mat = torch.tensor(base_segment_indexes)
        curr_clustered_segments = torch.where(segment_indexes_mat >= self.history_buffer_seg_end)[0]

        # Check if the current buffer result is falling short compared to `self.current_n`.
        if emb_in[curr_clustered_segments].shape[0] < self.current_n:
            delta_count = self.current_n - emb_in[curr_clustered_segments].shape[0]
            fill_in_emb = torch.tile(emb_in[curr_clustered_segments][-1], (delta_count, 1))
            emb_curr = torch.vstack((emb_in[curr_clustered_segments], fill_in_emb))
        else:
            emb_curr = emb_in[curr_clustered_segments]
        return emb_curr

    def reduce_embedding_sets(
        self, 
        emb_in: torch.Tensor, 
        base_segment_indexes: torch.Tensor
        ) -> Tuple[torch.Tensor, bool]:
        """
        Merge the given embedding vectors based on the calculate affinity matrix.

        Args:
            emb_in (Tensor):
                If self.isOnline is False:
                    `emb` contains only current speaker embedding inputs, which is FIFO queue
                If self.isOnline is True:
                    `emb` contains history buffer and FIFO queue
            base_segment_indexes (Tensor):
                Tensor containing unique segment (embedding vector) index

        Returns:
            history_embedding_buffer_emb (Tensor):
                Matrix containing merged embedding vectors of the previous frames.
                This matrix is referred to as "history buffer" in this class.
            update_speaker_register (bool):
                Boolean indicates whether to update speaker

        Example:

            at the frame index where `isOnline` turns to True:

            |---hist-buffer---|-----FIFO-queue-----|

            self.history_n = 10
            self.current_n = 20

            Step (1)
            |-----------------|ABCDEF--------------|

            If we get two more segments, "NN" as in the description:
            history buffer = 10
            current buffer = 22

            Step (2)
            |-----------------|ABCDEF--------------XY|

            The newly accepted embeddings go through a FIFO queue (first embedding, first merged)
            history buffer = 12
            current buffer = 20

            Step (3)
            |-----------------AB|CDEF--------------XY|

            After merging (reducing) the embedding set:
            history buffer = 10
            current buffer = 20

            Step (4)
            |================|CDEF--------------XY|

            After clustering:

            |0000011111|11110000110010010011|

            This label is self.Y_fullhist (shape is (history_n + current_n) )

        self.history_buffer_seg_end (int):
            The total number of segments that have been merged from the beginning of the session.
            (=hist_curr_boundary)

        """
        update_speaker_register, new_emb_n, pre_embs = self.prepare_embedding_update(emb_in, base_segment_indexes)

        # Update the history/current_buffer boundary cursor
        total_emb, total_cluster_labels = [], []

        if update_speaker_register:

            # Calculate how many embedding vectors should be reduced per speaker
            class_target_vol = get_merge_quantity(
                num_to_be_removed=new_emb_n,
                pre_clus_labels=self.pre_clus_labels,
                min_count_per_cluster=self._minimum_segments_per_buffer,
            )

            # Merge the segments in the history buffer
            for spk_idx, target_num in enumerate(list(class_target_vol)):
                merged_embs, merged_clus_labels, _ = run_reducer(
                    pre_embs=pre_embs,
                    target_spk_idx=spk_idx,
                    merge_quantity=target_num,
                    pre_clus_labels=self.pre_clus_labels,
                )
                total_emb.append(merged_embs)
                total_cluster_labels.append(merged_clus_labels)

            self.history_embedding_buffer_emb = torch.vstack(total_emb)
            self.history_embedding_buffer_label = torch.hstack(total_cluster_labels)
            if self.history_embedding_buffer_emb.shape[0] != self.history_n:
                raise ValueError("History embedding size is not maintained correctly.")

        else:
            total_emb.append(self.history_embedding_buffer_emb)
            total_cluster_labels.append(self.history_embedding_buffer_label)

        # `emb_curr` is the incumbent set of embeddings which is the the latest.
        emb_curr = self.make_constant_length_emb(emb_in, base_segment_indexes)
        total_emb.append(emb_curr)

        # Before perform clustering, we attach the current_n number of estimated speaker labels
        # from the previous clustering result.
        total_cluster_labels.append(self.Y_fullhist[-self.current_n :])

        history_and_current_emb = torch.vstack(total_emb)
        history_and_current_labels = torch.hstack(total_cluster_labels)
        if history_and_current_emb.shape[0] != len(history_and_current_labels):
            raise ValueError("history_and_current_emb has a mismatch with history_and_current_labels.")

        self.max_embed_count = max(self.total_segments_processed_count, self.max_embed_count)
        return history_and_current_emb, update_speaker_register

    def get_reduced_mat(self, emb_in, base_segment_indexes) -> Tuple[torch.Tensor, bool]:
        """
        Choose whether we want to add embeddings to the memory or not.
        The processing buffer has size of (self.current_n + self.history_n).

        1. If margin_seg_n > 0, this means we have more embedding vectors than we can hold in the processing buffer.
            - `isOnline` should be `True`
            - reduce the number of embedding vectors by merging the closest ones.
                call `self.reduce_embedding_sets` function

        2. If margin_seg_n <= 0, this means that we can accept more embedding vectors and yet to fill the processing buffer.
            - `isOnline` should be `False`
            - We replace `merged_emb` variable with the raw input `emb_in`.
            - `add_new` is `True`, since we are adding more embedding vectors to `merged_emb` variable.

        Args:
            emb_in (Tensor):
                If self.isOnline is False:
                    `emb` contains only current speaker embedding inputs
            base_segment_indexes (Tensor):
                Tensor containing unique segment (embedding vector) index

        Returns:
            merged_emb (Tensor):
                Matrix containing merged embedding vectors of the previous frames.
                This matrix is referred to as "history buffer" in this class.
            add_new (bool):
                Boolean that indicates whether there is a new set of segments. Depending on the VAD timestamps,
                the number of subsegments can be ocassionally decreased. If `add_new=True`, then it adds the newly
                acquired cluster labels.

        """
        margin_seg_n = emb_in.shape[0] - (self.current_n + self.history_n)
        if margin_seg_n > 0:
            self.isOnline = True
            merged_emb, add_new = self.reduce_embedding_sets(emb_in, base_segment_indexes)
        else:
            self.isOnline = False
            merged_emb = emb_in
            add_new = True
        return merged_emb, add_new

    def match_labels(self, Y_new: torch.Tensor, add_new: bool) -> torch.Tensor:
        """
        self.history_buffer_seg_end is a timestamp that tells to which point is history embedding contains from self.Y_fullhist.
        If embedding reducing is done correctly, we should discard  (0, self.history_n) amount and take
        (self.history_n, len(Y_new) ) from the new clustering output Y_new.

        Args:
            Y_new (Tensor):
                The newly generated clustering label sequence that may have different permutations with the existing
                speaker labels in the history buffer.
            add_new (bool):
                This variable indicates whether there is a new set of segments. Depending on the VAD timestamps,
                the number of subsegments can be ocassionally decreased. If `add_new=True`, then it adds the newly
                acquired cluster labels.

        Returns:
            Y_out (Tensor):
                Permutation-matched speaker labels based on history buffer
        """
        if self.isOnline:
            # Online clustering mode with history buffer
            Y_old = torch.hstack((self.history_embedding_buffer_label, self.Y_fullhist[self.history_buffer_seg_end :]))

            # Stitch the old history and new cluster labels
            Y_matched = stitch_cluster_labels(Y_old=Y_old, Y_new=Y_new, with_history=True).to(self.device)

            if add_new:
                if Y_matched[self.history_n :].shape[0] != self.current_n:
                    raise ValueError("Update point sync is not correct.")
                # Concatenate the newly generated speaker labels
                Y_out = torch.hstack((self.Y_fullhist[: self.history_buffer_seg_end], Y_matched[self.history_n :]))
                self.Y_fullhist = Y_out
            else:
                # Do not update cumulative labels since there are no new segments.
                Y_out = self.Y_fullhist[: Y_new.shape[0]]
        else:
            # If no memory is used, offline clustering is applied.
            Y_out = stitch_cluster_labels(Y_old=self.Y_fullhist, Y_new=Y_new, with_history=False).to(self.device)
            self.Y_fullhist = Y_out
        return Y_out

    def forward_infer(self, 
        emb: torch.Tensor, 
        frame_index: int, 
        cuda: bool, 
        device: torch.device) -> torch.Tensor:
        """
        Perform speaker clustering in online mode. Embedding vector set `emb` is expected to be containing
        history embeddings to count the number of speakers.

        Args:
            emb (Tensor):
                If self.isOnline is False:
                    `emb` contains only current speaker embedding inputs
                If self.isOnline is True:
                    `emb` is a concatenated matrix with history embedding and current embedding inputs
            frame_index (int):
                Unique index for each segment (also each embedding vector)
            cuda (bool):
                Boolean that determines whether cuda is used or not
            device (torch.device):
                `torch.device` variable

        Returns:
            Y (Tensor):
                Speaker labels for history embeddings and current embedding inputs
        """
        mat = getCosAffinityMatrix(emb)
        if emb.shape[0] == 1:
            Y = torch.zeros((1,), dtype=torch.int32)

        else:
            nmesc = NMESC(
                mat,
                max_num_speakers=self.max_num_speakers,
                max_rp_threshold=self.max_rp_threshold,
                sparse_search=True,
                maj_vote_spk_count=False,
                sparse_search_volume=self.sparse_search_volume,
                fixed_thres=self.fixed_thres,
                nme_mat_size=256,
                device=device,
            )

            est_num_of_spk, affinity_mat = self.online_spk_num_estimation(mat, nmesc, frame_index)
            spectral_model = SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda, device=device)
            Y = spectral_model.forward(affinity_mat)
        return Y
