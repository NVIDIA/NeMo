from sklearn.cluster import SpectralClustering as sklearn_SpectralClustering
import numpy as np
from tqdm import tqdm

def get_kneighbors_conn(X_dist, p_neighbors):
    X_dist_out = np.zeros_like(X_dist)
    for i, line in enumerate(X_dist):
        sorted_idx = np.argsort(line)
        sorted_idx = sorted_idx[::-1]
        indices = sorted_idx[:p_neighbors]
        print(X_dist_out.shape)
        print(indices.shape)
        X_dist_out[indices, i] = 1
    return X_dist_out

def get_X_conn_from_dist(X_dist_raw, p_neighbors):
    # p_neighbors = int(X_dist_raw.shape[0] * threshold)
    X_r = get_kneighbors_conn(X_dist_raw, p_neighbors) 
    X_conn_from_dist= 0.5 * (X_r + X_r.T)
    return X_conn_from_dist

def NMEanalysis(mat, SPK_MAX, max_rp_threshold=0.250, sparse_search=True, search_p_volume=20, fixed_thres=None):
        eps = 1e-10
        eig_ratio_list = []
        if fixed_thres:
            p_neighbors_list = [ int(mat.shape[0] * fixed_thres) ]
            max_N = p_neighbors_list[0]
        else:
            max_N = int(mat.shape[0] * max_rp_threshold)
            if sparse_search:
                N = min(max_N, search_p_volume)
                p_neighbors_list = list(np.linspace(1, max_N, N, endpoint=True).astype(int))
            else:
                p_neighbors_list = list(range(1, max_N))
            print("Scanning eig_ratio of length [{}] mat size [{}] ...".format(len(p_neighbors_list), mat.shape[0]))
        
        est_spk_n_dict = {}
        for p_neighbors in tqdm(p_neighbors_list):
            X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors)
            est_num_of_spk, lambdas, lambda_gap_list = estimate_num_of_spkrs(X_conn_from_dist, SPK_MAX)
            est_spk_n_dict[p_neighbors] = (est_num_of_spk, lambdas)
            arg_sorted_idx = np.argsort(lambda_gap_list[:SPK_MAX])[::-1] 
            max_key = arg_sorted_idx[0]  
            max_eig_gap = lambda_gap_list[max_key]/(max(lambdas) + eps) 
            eig_ratio_value = (p_neighbors/mat.shape[0])/(max_eig_gap+eps)
            eig_ratio_list.append(eig_ratio_value)
         
        index_nn = np.argmin(eig_ratio_list)
        rp_p_neighbors = p_neighbors_list[index_nn]
        X_conn_from_dist = get_X_conn_from_dist(mat, rp_p_neighbors)
        if not isFullyConnected(X_conn_from_dist):
            X_conn_from_dist, rp_p_neighbors = gc_thres_min_gc(mat, max_N, p_neighbors_list)
        
        return X_conn_from_dist, float(rp_p_neighbors/mat.shape[0]), est_spk_n_dict[rp_p_neighbors][0], est_spk_n_dict[rp_p_neighbors][1], rp_p_neighbors

class store_variables():
    pass

def COSclustering(key, mat, mat_spkcount):
    param=store_variables()
    param.max_speaker=2
    param.spt_est_thres='NMESC'
    param.threshold='None'
    param.reco2num_spk='None'

    est_num_spks_out_list=[]
    nmesc_thres_list=[]
    X_dist_raw = mat
    rp_threshold = param.threshold
    if param.spt_est_thres in ["EigRatio", "NMESC"] or param.threshold == "EigRatio":
        # param.sparse_search = False
        print("Running NME-SC and estimating the number of speakers...")
        X_conn_spkcount, rp_thres_spkcount, est_num_of_spk, lambdas, p_neigh_spkcount = NMEanalysis(mat_spkcount, param.max_speaker)
        rp_threshold = rp_thres_spkcount 
        nmesc_thres_list.append("{} {:2.3f}".format(key, rp_thres_spkcount))
            
    
    ### X_conn_from_dist is used for actual clustering results. 
    if mat.shape[0] != mat_spkcount.shape[0]:
        p_neigh = int(p_neigh_spkcount * (mat.shape[0]/mat_spkcount.shape[0]))
    else:
        p_neigh = p_neigh_spkcount

    X_conn_from_dist = get_X_conn_from_dist(mat, p_neigh)

    '''
    Determine the number of speakers.
    if param.reco2num_spk contains speaker number info, we use that.
    Otherwise we estimate the number of speakers using estimate_num_of_spkrs()

    '''
    if param.reco2num_spk != 'None': 
        est_num_of_spk = int(reco2num_dict[key])
        ### Use the given number of speakers
        est_num_of_spk = min(est_num_of_spk, param.max_speaker) 
        # _, lambdas, lambda_gap_list = estimate_num_of_spkrs(X_conn_from_dist, param.max_speaker)
        print_status_givenNspk(key, mat, rp_threshold, est_num_of_spk, param)

    else: 
        ### Estimate the number of speakers in the given session
        print_status_estNspk(key, mat, rp_threshold, est_num_of_spk, param)

    

    est_num_spks_out_list.append( [key, str(est_num_of_spk)] ) 
    
    ### Handle the sklearn/numpy bug of eigenvalue parameter.
    # ipdb.set_trace()
    spectral_model = sklearn_SpectralClustering(affinity='precomputed', 
                                            eigen_solver='amg',
                                            random_state=0,
                                            n_jobs=3, 
                                            n_clusters=est_num_of_spk,
                                            eigen_tol=1e-10)
    
    Y = spectral_model.fit_predict(X_conn_from_dist)

    return Y

@staticmethod 
def print_status_estNspk(key, mat, rp_threshold, est_num_of_spk, param):
    # if param.threshold != 'None':
        # rp_threshold = float(param.threshold)
    print(key, " score_metric:", param.score_metric, 
                    " affinity matrix pruning - threshold: {:3.3f}".format(rp_threshold),
                    " key:", key,"Est # spk: " + str(est_num_of_spk), 
                    " Max # spk:", param.max_speaker, 
                    " MAT size : ", mat.shape)


@staticmethod 
def print_status_givenNspk(key, mat, rp_threshold, est_num_of_spk, param):
    # if param.threshold != 'None'
        # rp_threshold = float(param.threshold)
    print(" score_metric:", param.score_metric,
                    " Rank based pruning - RP threshold: {:4.4f}".format(rp_threshold), 
                    " key:", key,
                    " Given Number of Speakers (reco2num_spk): " + str(est_num_of_spk), 
                    " MAT size : ", mat.shape)