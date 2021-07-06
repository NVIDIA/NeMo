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

import warnings

import numpy as np
import scipy
import torch
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))


def isGraphFullyConnected(affinity_mat):
    return getTheLargestComponent(affinity_mat, 0).sum() == affinity_mat.shape[0]


def getTheLargestComponent(affinity_mat, seg_index):
    """
    Find the largest affinity_mat connected components for each given node.
    This is for checking whether the affinity_mat is fully connected.
    """
    num_of_segments = affinity_mat.shape[0]

    connected_nodes = np.zeros(num_of_segments).astype(np.bool)
    nodes_to_explore = np.zeros(num_of_segments).astype(np.bool)

    nodes_to_explore[seg_index] = True
    for k in range(num_of_segments):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            neighbors = affinity_mat[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def getKneighborsConnections(affinity_mat, p_value):
    """
    For a given p_value value, binarize the given affinity matrix.
    """
    X_dist_out = np.zeros_like(affinity_mat)
    for i, line in enumerate(affinity_mat):
        sorted_idx = np.argsort(line)
        sorted_idx = sorted_idx[::-1]
        indices = sorted_idx[:p_value]
        X_dist_out[indices, i] = 1
    return X_dist_out


def getAffinityGraphMat(X_dist_raw, p_value):
    X_r = getKneighborsConnections(X_dist_raw, p_value)
    X_conn_from_dist = 0.5 * (X_r + X_r.T)
    return X_conn_from_dist


def getMinimumConnection(mat, max_N, n_list):
    """
    Generates connections until fully connect all the nodes in the graph.
    If graph is not fully connected, it might generate an inaccurate results.
    """
    p_value, index = 1, 0
    affinity_mat = getAffinityGraphMat(mat, p_value)
    fully_connected = isGraphFullyConnected(affinity_mat)
    for i, p_value in enumerate(n_list):
        fully_connected = isGraphFullyConnected(affinity_mat)
        affinity_mat = getAffinityGraphMat(mat, p_value)
        if fully_connected or p_value > max_N:
            break

    return affinity_mat, p_value


def getCosAffinityMatrix(emb):
    sim_d = cosine_similarity(emb)
    scaler.fit(sim_d)
    sim_d = scaler.transform(sim_d)
    return sim_d


def getLaplacian(X):
    X[np.diag_indices(X.shape[0])] = 0
    A = X
    D = np.sum(np.abs(A), axis=1)
    D = np.diag(D)
    L = D - A
    return L


def eigDecomposeGPU(Laplacian):
    device = torch.cuda.current_device()
    laplacian_torch = torch.from_numpy(Laplacian).float().to(device)
    lambdas_torch, _ = torch.linalg.eigh(laplacian_torch)
    lambdas = lambdas_torch.cpu().numpy()
    return lambdas


def eigDecomposeCPU(Laplacian):
    lambdas, _ = scipy.linalg.eigh(Laplacian)
    return lambdas


class _SpectralClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, random_state=0, n_init=10, p_value=10, n_jobs=None, cuda=False):

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.p_value = p_value
        self.affinity_matrix_ = None
        self.cuda = cuda

    def fit(self, X):
        if X.shape[0] != X.shape[1]:
            raise ValueError("The affinity matrix is not square matrix.")

        self.affinity_matrix_ = X
        self.labels_ = self.clusterSpectralEmbeddings(self.affinity_matrix_, n_init=self.n_init, cuda=self.cuda)
        return self

    def clusterSpectralEmbeddings(self, affinity, n_init=10, cuda=False):

        spectral_emb = self.getSpectralEmbeddings(affinity, n_spks=self.n_clusters, drop_first=False, cuda=cuda)

        _, labels, _ = k_means(spectral_emb, self.n_clusters, random_state=self.random_state, n_init=n_init)
        return labels

    def getSpectralEmbeddings(self, affinity_mat, n_spks=8, drop_first=True, cuda=False):

        n_nodes = affinity_mat.shape[0]
        if not isGraphFullyConnected(affinity_mat):
            warnings.warn("Graph is not fully connected and clustering result might not be accurate.")

        Laplacian = getLaplacian(affinity_mat)
        if cuda:
            device = torch.cuda.current_device()
            laplacian_torch = torch.from_numpy(Laplacian).float().to(device)
            lambdas_torch, diffusion_map_torch = torch.linalg.eigh(laplacian_torch)
            lambdas = lambdas_torch.cpu().numpy()[:n_spks]
            diffusion_map = diffusion_map_torch.cpu().numpy()[:, :n_spks]
        else:
            lambdas_, diffusion_map_ = scipy.linalg.eigh(Laplacian)
            lambdas = lambdas_[:n_spks]
            diffusion_map = diffusion_map_[:, :n_spks]

        embedding = diffusion_map.T[n_spks::-1]
        return embedding[:n_spks].T


class NMESC:
    def __init__(
        self,
        mat,
        max_num_speaker=10,
        max_rp_threshold=0.250,
        sparse_search=True,
        sparse_search_volume=30,
        use_subsampling_for_NME=True,
        fixed_thres=None,
        cuda=False,
        NME_mat_size=512,
    ):
        """
        Normalized Maximum Eigengap based Spectral Clustering (NME-SC)
        Normalized Eigengap analysis to get p-value for affinity binarization
        and estimated number of speakers.

        p_value (also referred to as p_neighbors) means we take
        top p number of affinity values
        and convert those to 1.
        The other affinity values are converted to zero.

        Reference: https://arxiv.org/abs/2003.02405
        """
        self.max_num_speaker = max_num_speaker
        self.max_rp_threshold = max_rp_threshold
        self.use_subsampling_for_NME = use_subsampling_for_NME
        self.NME_mat_size = NME_mat_size
        self.sparse_search = sparse_search
        self.sparse_search_volume = sparse_search_volume
        self.fixed_thres = fixed_thres
        self.cuda = cuda
        self.eps = 1e-10
        self.max_N = None
        self.mat = mat
        self.p_value_list = []

    def NMEanalysis(self):
        if self.use_subsampling_for_NME:
            subsample_ratio = self.subsampleAffinityMat(self.NME_mat_size)

        self.p_value_list = self.getPvalueList()

        eig_ratio_list, est_spk_n_dict = [], {}
        for p_value in self.p_value_list:
            est_num_of_spk, g_p = self.getEigRatio(p_value)
            est_spk_n_dict[p_value] = est_num_of_spk
            eig_ratio_list.append(g_p)

        index_nn = np.argmin(eig_ratio_list)
        rp_p_value = self.p_value_list[index_nn]
        affinity_mat = getAffinityGraphMat(self.mat, rp_p_value)

        if not isGraphFullyConnected(affinity_mat):
            affinity_mat, rp_p_value = getMinimumConnection(self.mat, self.max_N, self.p_value_list)

        p_hat_value = int(subsample_ratio * rp_p_value)
        est_num_of_spk = est_spk_n_dict[rp_p_value]
        return est_num_of_spk, p_hat_value

    def subsampleAffinityMat(self, NME_mat_size):
        """
        Performs Subsampling of affinity matrix.
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
                Targeted matrix size.

        Returns:
            subsample_ratio : (float)
                The ratio between NME_mat_size and the original matrix size.

        """
        subsample_ratio = max(1, self.mat.shape[0] / NME_mat_size)
        if int(subsample_ratio) != 1:
            self.mat = self.mat[:: int(subsample_ratio), :: int(subsample_ratio)]
        return subsample_ratio

    def getEigRatio(self, p_neighbors):
        """
        For a given p_neighbors value,
        calculates g_p, which is a ratio
        between p_neighbors and the maximum eigengap.
        For more details: https://arxiv.org/abs/2003.02405

        Parameters:
            p_neighbors: (int)
                Determins how many binary graph connections we want to keep for each row.

        Returns:
            est_num_of_spk: (int)
                Estimated number of speakers

            g_p: (float)
                The ratio between p_neighbors value and the maximum eigen gap value.
        """

        affinity_mat = getAffinityGraphMat(self.mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = self.estimateNumofSpeakers(affinity_mat)
        arg_sorted_idx = np.argsort(lambda_gap_list[: self.max_num_speaker])[::-1]
        max_key = arg_sorted_idx[0]
        max_eig_gap = lambda_gap_list[max_key] / (max(lambdas) + self.eps)
        g_p = (p_neighbors / self.mat.shape[0]) / (max_eig_gap + self.eps)

        return est_num_of_spk, g_p

    def getPvalueList(self):
        """Generates a p-value (p_neighbour) list for searching.
        """
        if self.fixed_thres:
            p_value_list = [int(self.mat.shape[0] * self.fixed_thres)]
            self.max_N = p_value_list[0]
        else:
            self.max_N = int(self.mat.shape[0] * self.max_rp_threshold)
            if self.sparse_search:
                N = min(self.max_N, self.sparse_search_volume)
                p_value_list = list(np.linspace(1, self.max_N, N, endpoint=True).astype(int))
            else:
                p_value_list = list(range(1, self.max_N))

        return p_value_list

    def estimateNumofSpeakers(self, X_conn):
        """Estimate the number of speakers using eigen decompose on Laplacian Matrix.
        """
        Laplacian = getLaplacian(X_conn)
        if self.cuda:
            lambdas = eigDecomposeGPU(Laplacian)
        else:
            lambdas = eigDecomposeCPU(Laplacian)

        lambdas = np.sort(lambdas)
        lambda_gap_list = self.getLamdaGaplist(lambdas)
        num_of_spk = np.argmax(lambda_gap_list[: min(self.max_num_speaker, len(lambda_gap_list))]) + 1
        return num_of_spk, lambdas, lambda_gap_list

    def getLamdaGaplist(self, lambdas):
        lambdas = np.real(lambdas)
        return list(lambdas[1:] - lambdas[:-1])


def COSclustering(key, emb, oracle_num_speakers=None, max_num_speaker=8, min_samples=6, cuda=False):
    """
    Clustering method for speaker diarization based on cosine similarity.

    Parameters:
        key: (str)
            Speaker uniq name
        emb: (numpy array)
            Speaker embedding
        oracle_num_speaker: (int or None)
            Oracle number of speakers if known else None
        max_num_speaker: (int)
            Maximum number of clusters to consider for each session
        min_samples: (int)
            Minimum number of samples required for NME clustering, this avoids
            zero p_neighbour_lists. Default of 6 is selected since  (1/rp_threshold) >= 4.

    Returns:
        Y: (List[int])
            Speaker label for each segment.
    """
    mat = getCosAffinityMatrix(emb)
    if oracle_num_speakers:
        max_num_speaker = oracle_num_speakers

    if emb.shape[0] > min_samples:
        nmesc = NMESC(
            mat,
            max_num_speaker=max_num_speaker,
            max_rp_threshold=0.25,
            sparse_search=True,
            sparse_search_volume=30,
            fixed_thres=None,
            NME_mat_size=300,
            cuda=cuda,
        )

        est_num_of_spk, p_hat_value = nmesc.NMEanalysis()
        affinity_mat = getAffinityGraphMat(mat, p_hat_value)
    else:
        affinity_mat = mat

    if oracle_num_speakers:
        est_num_of_spk = oracle_num_speakers

    spectral_model = _SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda)
    Y = spectral_model.fit_predict(affinity_mat)

    return Y
