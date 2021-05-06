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


import numpy as np
import scipy
from scipy import sparse
from sklearn.cluster import SpectralClustering as sklearn_SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))


def get_kneighbors_conn(X_dist, p_neighbors):
    X_dist_out = np.zeros_like(X_dist)
    for i, line in enumerate(X_dist):
        sorted_idx = np.argsort(line)
        sorted_idx = sorted_idx[::-1]
        indices = sorted_idx[:p_neighbors]
        X_dist_out[indices, i] = 1
    return X_dist_out


def get_X_conn_from_dist(X_dist_raw, p_neighbors):
    X_r = get_kneighbors_conn(X_dist_raw, p_neighbors)
    X_conn_from_dist = 0.5 * (X_r + X_r.T)
    return X_conn_from_dist


def isFullyConnected(X_conn_from_dist):
    """ Return whether the graph is connected (True) or Not (False).
    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.
    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not.
    """
    gC = _graph_connected_component(X_conn_from_dist, 0).sum() == X_conn_from_dist.shape[0]
    return gC


def gc_thres_min_gc(mat, max_n, n_list):
    p_neighbors, index = 1, 0
    X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors)
    fully_connected = isFullyConnected(X_conn_from_dist)
    for i, p_neighbors in enumerate(n_list):
        fully_connected = isFullyConnected(X_conn_from_dist)
        X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors)
        if fully_connected or p_neighbors > max_n:
            break

    return X_conn_from_dist, p_neighbors


def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node.
    Parameters
    ----------
    graph : array-like of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.
    node_id : int
        The index of the query node of the graph.
    Returns
    -------
    connected_components_matrix : array-like of shape (n_samples,)
        An array of bool value indicating the indexes of the nodes
        belonging to the largest connected components of the given query
        node.
    """
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def getLaplacian(X):
    """
    returns laplacian matrix of X
    """
    X[np.diag_indices(X.shape[0])] = 0
    A = X
    D = np.sum(np.abs(A), axis=1)
    D = np.diag(D)
    L = D - A
    return L


def eig_decompose(L, k):
    try:
        lambdas, eig_vecs = scipy.linalg.eigh(L)
    except:
        try:
            lambdas = scipy.linalg.eigvals(L)
            eig_vecs = None
        except:
            lambdas, eig_vecs = scipy.sparse.linalg.eigsh(L)  ### Inaccurate results
    return lambdas, eig_vecs


def getLamdaGaplist(lambdas):
    lambda_gap_list = []
    for i in range(len(lambdas) - 1):
        lambda_gap_list.append(float(lambdas[i + 1]) - float(lambdas[i]))
    return lambda_gap_list


def estimate_num_of_spkrs(X_conn, SPK_MAX):
    """
    Estimate number of speakers using eigen decompose on Laplacian Matrix
    """
    L = getLaplacian(X_conn)
    lambdas, eig_vals = eig_decompose(L, k=X_conn.shape[0])
    lambdas = np.sort(lambdas)
    lambda_gap_list = getLamdaGaplist(lambdas)
    num_of_spk = np.argmax(lambda_gap_list[: min(SPK_MAX, len(lambda_gap_list))]) + 1
    return num_of_spk, lambdas, lambda_gap_list


def NMEanalysis(mat, SPK_MAX, max_rp_threshold=0.250, sparse_search=True, search_p_volume=20, fixed_thres=None):
    """
    Normalized Eigengap analysis to get p_neighbours and estimated num of speakers
    Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8937487&tag=1 
    """
    eps = 1e-10
    eig_ratio_list = []
    if fixed_thres:
        p_neighbors_list = [int(mat.shape[0] * fixed_thres)]
        max_N = p_neighbors_list[0]
    else:
        max_N = int(mat.shape[0] * max_rp_threshold)
        if sparse_search:
            N = min(max_N, search_p_volume)
            p_neighbors_list = list(np.linspace(1, max_N, N, endpoint=True).astype(int))
        else:
            p_neighbors_list = list(range(1, max_N))

    est_spk_n_dict = {}
    for p_neighbors in p_neighbors_list:
        X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = estimate_num_of_spkrs(X_conn_from_dist, SPK_MAX)
        est_spk_n_dict[p_neighbors] = (est_num_of_spk, lambdas)
        arg_sorted_idx = np.argsort(lambda_gap_list[:SPK_MAX])[::-1]
        max_key = arg_sorted_idx[0]
        max_eig_gap = lambda_gap_list[max_key] / (max(lambdas) + eps)
        eig_ratio_value = (p_neighbors / mat.shape[0]) / (max_eig_gap + eps)
        eig_ratio_list.append(eig_ratio_value)

    index_nn = np.argmin(eig_ratio_list)
    rp_p_neighbors = p_neighbors_list[index_nn]
    X_conn_from_dist = get_X_conn_from_dist(mat, rp_p_neighbors)
    if not isFullyConnected(X_conn_from_dist):
        X_conn_from_dist, rp_p_neighbors = gc_thres_min_gc(mat, max_N, p_neighbors_list)

    return (
        X_conn_from_dist,
        float(rp_p_neighbors / mat.shape[0]),
        est_spk_n_dict[rp_p_neighbors][0],
        est_spk_n_dict[rp_p_neighbors][1],
        rp_p_neighbors,
    )


def get_eigen_matrix(emb):
    """
    input
    emb: embedding
    
    output
    similarity matrix
    """
    sim_d = cosine_similarity(emb)
    scaler.fit(sim_d)
    sim_d = scaler.transform(sim_d)

    return sim_d


def COSclustering(key, emb, oracle_num_speakers=None, max_num_speaker=8):
    """
    input:
    key (str): speaker uniq name
    emb (np array): speaker embedding
    oracle_num_speaker (int or None): oracle number of speakers if known else None
    max_num_speakers (int): maximum number of clusters to consider for each session
    output:
    Y (List[int]): speaker labels
    """
    est_num_spks_out_list = []
    mat = get_eigen_matrix(emb)

    if oracle_num_speakers:
        max_num_speaker = oracle_num_speakers

    X_conn_spkcount, rp_thres_spkcount, est_num_of_spk, lambdas, p_neigh = NMEanalysis(mat, max_num_speaker)
    X_conn_from_dist = get_X_conn_from_dist(mat, p_neigh)

    if oracle_num_speakers:
        est_num_of_spk = oracle_num_speakers

    est_num_spks_out_list.append([key, str(est_num_of_spk)])

    # Perform spectral clustering
    spectral_model = sklearn_SpectralClustering(
        affinity='precomputed',
        eigen_solver='amg',
        random_state=0,
        n_jobs=-1,
        n_clusters=est_num_of_spk,
        eigen_tol=1e-10,
    )

    Y = spectral_model.fit_predict(X_conn_from_dist)

    return Y
