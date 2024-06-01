import os
import torch
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import os.path as osp
import scipy.sparse as sp
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
from rich import print
from rich.progress import track, Progress
from contextlib import redirect_stdout
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


def compute_matching(labels_true, labels_pred):
    unique_labels_true = np.unique(labels_true)
    unique_labels_pred = np.unique(labels_pred)

    similarity_matrix = np.zeros((len(unique_labels_true), len(unique_labels_pred)))

    for i, true_label in enumerate(unique_labels_true):
        true_mask = labels_true == true_label
        for j, pred_label in enumerate(unique_labels_pred):
            pred_mask = labels_pred == pred_label
            similarity_matrix[i, j] = np.sum(true_mask & pred_mask)

    row_ind, col_ind = linear(-similarity_matrix)

    label_mapping = {
        unique_labels_pred[col]: unique_labels_true[row] for row, col in zip(row_ind, col_ind)
    }

    matched_labels_true = np.array([label_mapping[label] for label in labels_pred])

    return similarity_matrix, matched_labels_true, label_mapping

def evaluate_clustering(labels_true, labels_pred):
    similarity_matrix, matched_labels_true, label_mapping = compute_matching(
        labels_true, labels_pred
    )

    ari = ari_score(labels_true, matched_labels_true)
    nmi = nmi_score(labels_true, matched_labels_true)
    f1 = metrics.f1_score(labels_true, matched_labels_true, average='weighted')
    acc = metrics.accuracy_score(labels_true, matched_labels_true)

    return similarity_matrix, matched_labels_true, label_mapping, ari, nmi, f1, acc

def eva(labels_true, labels_pred, out_file=None):
    similarity_matrix, matched_labels_true, label_mapping, ari, nmi, f1, acc = evaluate_clustering(
        labels_true=labels_true, labels_pred=labels_pred
    )
    
    unique_labels_true = np.unique(labels_true)
    unique_labels_pred = np.unique(labels_pred)

    if out_file:
        with open(out_file, 'w') as f:
            with redirect_stdout(f):
                print(">>> Similarity Matrix:")
                header = "\t".join([""] + [f"Pred: {label}" for label in unique_labels_pred])
                print(header)
                for i, true_label in enumerate(unique_labels_true):
                    row = "\t".join(
                        [f"True: {true_label}"] + [
                            str(similarity_matrix[i, j]) for j in range(len(unique_labels_pred))
                        ]
                    )
                    print(row)

                # print("\n>>> Matched Labels True:", matched_labels_true)
                print(">>> Label Mapping:", label_mapping)
                print(">>> ARI:", ari)
                print(">>> NMI:", nmi)
                print(">>> F1-score:", f1)
                print(">>> Accuracy:", acc)
    else:
        print(">>> Similarity Matrix:")
        header = "\t".join([""] + [f"Pred: {label}" for label in unique_labels_pred])
        print(header)
        for i, true_label in enumerate(unique_labels_true):
            row = "\t".join(
                [f"True: {true_label}"] + [
                    str(similarity_matrix[i, j]) for j in range(len(unique_labels_pred))
                ]
            )
            print(row)

        # print("\n>>> Matched Labels True:", matched_labels_true)
        print(">>> Label Mapping:", label_mapping)
        print(">>> ARI:", ari)
        print(">>> NMI:", nmi)
        print(">>> F1-score:", f1)
        print(">>> Accuracy:", acc)

def mclust_R(emb_pca, num_cluster, modelNames='EEE', random_seed=2020):
    np.random.seed(random_seed)
    robjects.r.library("mclust")

    numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(numpy2ri.numpy2rpy(emb_pca), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    ret = mclust_res
    ret = ret.astype('int')
    return ret

def clustering(emb, n_clusters=7, start=0.1, end=3.0, increment=0.01, refinement=False):
    pca = PCA(n_components=10, random_state=42) 
    embedding = pca.fit_transform(emb.copy())
    emb_pca = embedding
    
    mclust_ret = mclust_R(emb_pca, num_cluster=n_clusters)
    mclust_ret = mclust_ret

    return mclust_ret

def draw_sub_type_map(
        bi_type, 
        data_name,
        types_dic,
        node_embed,
        method,
        time_stamp,
        seed,
        fixed_clus_count=3,
        cluster_method='leiden',
        cluster_with_fix_reso=True,
        resolution=None,
        eval=True,
        region_col_name="NULL"
):
    outdir=osp.join('results/', data_name, time_stamp, method)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    generated_data_fold = osp.join('generated/', data_name)

    cell_types = pd.read_csv(osp.join(generated_data_fold, 'cell_types.csv')).iloc[:, 0]
    if eval:
        labels_true_full = np.array(
            pd.read_csv(osp.join(generated_data_fold, 'regions.csv'))[region_col_name].tolist()
        )
    increment=0.02
    for t in types_dic:
        if t == bi_type:
            type_id_list = cell_types[cell_types == t].index.to_list()
            if eval:
                labels_true = labels_true_full[type_id_list]
            if cluster_method == 'leiden':
                if resolution == None:
                    # Fixed subtype number
                    adata = ad.AnnData(node_embed[type_id_list])
                    sc.pp.neighbors(
                        adata, 
                        n_neighbors=20, 
                        n_pcs=0, 
                        key_added='STACCI', 
                        random_state=seed
                    )
                    for res in sorted(list(np.arange(0.02, 5, increment)), reverse=True):
                        sc.tl.leiden(
                            adata, 
                            resolution=res, 
                            neighbors_key='STACCI', 
                            random_state=seed
                        )
                        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
                        if count_unique_leiden == fixed_clus_count:
                            print('Resolution:', res)
                            break
                    eval_resolution = res
                    sc.tl.leiden(
                        adata, 
                        key_added="leiden", 
                        neighbors_key='STACCI', 
                        resolution=eval_resolution, 
                        random_state=seed
                    )
                    sc.tl.umap(adata, neighbors_key='STACCI', random_state=seed)
                    cluster_labels = np.array(adata.obs['leiden'])
                    txt_lines = [ [idx, int(x)] for idx, x in zip(type_id_list, cluster_labels) ]
                    np.savetxt(
                        osp.join(
                            outdir, 
                            f"fixed_n={fixed_clus_count}_{t.replace('/', 'or')}_types.txt"
                        ), 
                        np.array(txt_lines), 
                        fmt='%3d', 
                        delimiter='\t'
                    )
                    draw_default_outdir(
                        generated_data_fold, 
                        outdir, 
                        f"STACCI_{t}", 
                        f"fixed_n={fixed_clus_count}_spatial_{t.replace('/', 'or')}", 
                        f"fixed_n={fixed_clus_count}_{t.replace('/', 'or')}_types.txt"
                    )

                    if eval:
                        labels_pred = [label for idx, label in txt_lines]
                        eva(labels_true, labels_pred, out_file=osp.join(outdir, 'leiden.txt'))

                if cluster_with_fix_reso:
                    # Fixed leiden resolution
                    adata = ad.AnnData(node_embed[type_id_list])
                    sc.pp.neighbors(
                        adata, 
                        n_neighbors=20, 
                        n_pcs=0, 
                        key_added='STACCI', 
                        random_state=seed
                    )
                    eval_resolution = 1 if resolution == None else resolution
                    sc.tl.leiden(
                        adata, 
                        key_added="leiden", 
                        neighbors_key='STACCI', 
                        resolution=eval_resolution, 
                        random_state=seed
                    )
                    sc.tl.umap(adata, neighbors_key='STACCI', random_state=seed)
                    cluster_labels = np.array(adata.obs['leiden'])
                    txt_lines = [ [idx, int(x)] for idx, x in zip(type_id_list, cluster_labels) ]
                    np.savetxt(
                        osp.join(
                            outdir, 
                            f"fixed_reso={eval_resolution}_{t.replace('/', 'or')}_types.txt"
                        ), 
                        np.array(txt_lines), 
                        fmt='%3d', 
                        delimiter='\t'
                    )
                    draw_default_outdir(
                        generated_data_fold, 
                        outdir, 
                        f"STACCI_{t}",
                        f"fixed_reso={eval_resolution}_spatial_{t.replace('/', 'or')}", 
                        f"fixed_reso={eval_resolution}_{t.replace('/', 'or')}_types.txt"
                    )

                    if eval:
                        labels_pred = [label for idx, label in txt_lines]
                        eva(
                            labels_true, 
                            labels_pred, 
                            out_file=osp.join(outdir, f'leiden_reso={eval_resolution}.txt')
                        )

            elif cluster_method == 'mclust': # Must cluster with fixed sub-type count
                emb = node_embed[type_id_list]
                ret = clustering(emb, n_clusters=fixed_clus_count)
                cell_cluster_type_list = ret - 1
                n_clusters = max(ret)
                title = f"STACCI_{t}_mclust"
                coordinates = np.load(
                    osp.join(generated_data_fold, 'coordinates.npy')
                )[type_id_list]

                txt_lines = [
                    [idx, int(x)] for idx, x in zip(type_id_list, cell_cluster_type_list)
                ]
                np.savetxt(
                    osp.join(outdir, f"mclust_fixed_n_{t.replace('/', 'or')}_types.txt"), 
                    np.array(txt_lines), 
                    fmt='%3d', 
                    delimiter='\t'
                )

                sc_cluster = plt.scatter(
                    x=coordinates[:, 0],
                    y=-coordinates[:, 1], 
                    s=5, 
                    c=cell_cluster_type_list, 
                    cmap='rainbow'
                )
                if n_clusters <= 3:
                    plt.legend(
                        *sc_cluster.legend_elements(), 
                        bbox_to_anchor=(1, 0.5), 
                        loc='center left', 
                        prop={'size': 9}
                    )
                else:
                    plt.legend(
                        *sc_cluster.legend_elements(num=n_clusters), 
                        bbox_to_anchor=(1, 0.5), 
                        loc='center left',
                        prop={'size': 9}
                    )
                plt.xticks([])
                plt.yticks([])
                plt.axis('scaled')
                plt.title(f'{title}')
                plt.savefig(
                    osp.join(outdir, f"mclust_fixed_n_spatial_{t.replace('/', 'or')}.png"), 
                    dpi=400, 
                    bbox_inches='tight'
                ) 
                plt.clf()
                
                if eval:
                    labels_pred = cell_cluster_type_list # TODO: can be saved with type_id_list
                    eva(labels_true, labels_pred, out_file=osp.join(outdir, 'mclust.txt'))
            else:
                pass

def draw_default_outdir(data_folder, save_path, title, fig_name, type_txt_file_name):
    if type_txt_file_name is None:
        f = open(osp.join(save_path, 'types.txt'))
    else:
        f = open(osp.join(save_path, type_txt_file_name))
    line = f.readline() # drop the first line  
    cell_cluster_type_list = []
    id_list = []

    while line: 
        tmp = line.split('\t')
        cell_id = int(tmp[0]) # index start is start from 0 here
        id_list.append(cell_id)
        cell_cluster_type = int(tmp[1].replace('\n', ''))
        cell_cluster_type_list.append(cell_cluster_type)
        line = f.readline()
    f.close() 
    n_clusters = max(cell_cluster_type_list) + 1 # start from 0
    print(f'{n_clusters} clusters in drawing...')
    coordinates = np.load(osp.join(data_folder, 'coordinates.npy'))[id_list]

    sc_cluster = plt.scatter(
        x=coordinates[:, 0], 
        y=-coordinates[:, 1], 
        s=5, 
        c=cell_cluster_type_list, 
        cmap='rainbow'
    )
    if n_clusters <= 3:
        plt.legend(
            *sc_cluster.legend_elements(), 
            bbox_to_anchor=(1, 0.5), 
            loc='center left', 
            prop={'size': 9}
        )
    else:
        plt.legend(
            *sc_cluster.legend_elements(num=n_clusters), 
            bbox_to_anchor=(1, 0.5), 
            loc='center left', 
            prop={'size': 9}
        )
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    plt.title(f'{title}')
    plt.savefig(osp.join(save_path, f'{fig_name}.png'), dpi=400, bbox_inches='tight') 
    plt.clf()

def find_range(nums, target):
    left = 0
    right = len(nums) - 1
    start = -1
    end = -1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            # 找到目标值，继续向左和向右搜索范围
            start = find_start(nums, target, left, mid)
            end = find_end(nums, target, mid, right)
            break
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1

    return [start, end]

def find_start(nums, target, left, right):
    index = -1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            index = mid
            right = mid - 1
        else:
            left = mid + 1

    return index

def find_end(nums, target, left, right):
    index = -1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            index = mid
            left = mid + 1
        else:
            right = mid - 1

    return index

def check_edge_in_adj(st, ed, sts, eds):
    st_st, st_ed = find_range(sts, st)
    if st_st == -1:
        return False
    if find_range(eds[st_st: st_ed], ed)[0] == -1:
        return False
    return True

def filter_attn_LRs(attn_LRs, adj, cut='FULL', return_std=False):
    sts, eds = adj.row, adj.col
    none_interactions = []
    all_false_interactions = []
    std_LRs = {}
    for interaction in track(attn_LRs.keys(), description='>>> Filtering...'):
        attn_LR = attn_LRs[interaction].tocoo()

        exists_mask = np.array(
            [check_edge_in_adj(st, ed, sts, eds) for st, ed in zip(attn_LR.row, attn_LR.col)]
        )

        if len(exists_mask) == 0:
            none_interactions.append(interaction)
            # std_LRs[interaction] = 0 # XXX: Should not assign if want to filter the voage interaction
            continue

        if not any(exists_mask):
            all_false_interactions.append(interaction)
            # std_LRs[interaction] = 0 # XXX: Should not assign if want to filter the voage interaction
            continue

        attn_LR_data_filtered = attn_LR.data[exists_mask]
        std_LRs[interaction] = np.std(np.concatenate([
            attn_LR_data_filtered, 
            np.zeros(adj.nnz - len(attn_LR_data_filtered))
        ]))
    print(f"#Interaction without edges={len(none_interactions)}")
    print(f"#Interaction without std={len(all_false_interactions)}")

    if cut == "FULL":
        ret_dict = dict(sorted(std_LRs.items(), key=lambda x: x[1], reverse=True))
    else:
        ret_dict = dict(sorted(std_LRs.items(), key=lambda x: x[1], reverse=True)[:cut])

    cut_attn_LRs = {key: attn_LRs[key].tocoo() for key in ret_dict.keys()}

    if return_std:
        return cut_attn_LRs, ret_dict
    return cut_attn_LRs

def replace_attn_LRs(attn_LRs, y, norm=False): # Notion: it is possible to find that the mat for certain interaction is empty after replacing!
    none_interactions_after_replace = []
    new_dict = {}
    for interaction in track(attn_LRs.keys(), description='>>> Replacing...'):
        attn_LR = attn_LRs[interaction]
        full_mat = attn_LR.toarray()

        min_val = float('inf')  # 初始化min_val为正无穷大
        max_val = float('-inf')  # 初始化max_val为负无穷大
        new_data = []
        for st, ed, e_val in zip(attn_LR.row, attn_LR.col, attn_LR.data):
            if y[st] == 1 and y[ed] == 0:
                new_val = full_mat[ed, st]
            else:
                new_val = e_val
            new_data.append(new_val)
            min_val = min(min_val, e_val)  # 更新最小值
            max_val = max(max_val, e_val)  # 更新最大值
        
        if len(new_data) == 0 or (max_val == 0 and min_val == 0): # If no edges or all zero, skip
            none_interactions_after_replace.append(interaction)
            continue
        if norm:
            if max_val != min_val:
                new_data = (np.array(new_data) - min_val) / (max_val - min_val)
            else: # if all the values of edges are the same, set them one
                new_data = np.ones_like(np.array(new_data))
        new_dict[interaction] = sp.coo_matrix(
            (new_data, (attn_LR.row, attn_LR.col)), 
            shape=attn_LR.shape, 
            dtype=attn_LR.dtype
        )
    print(f"#Interaction without edges after replacing={len(none_interactions_after_replace)}")
    return new_dict              

def _confirm_coo_mat(lst):
    if sp.isspmatrix(lst):
        return lst.tocoo()
    if not isinstance(lst, np.ndarray):
        lst = np.array(lst)
    return sp.coo_matrix(lst)

def _confirm_np_array(lst):
    if sp.isspmatrix(lst):
        return lst.toarray()
    if not isinstance(lst, np.ndarray):
        lst = np.array(lst)
    return lst

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

# 将字典保存到文件
def save_dict_to_file(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

# 从文件中加载字典
def load_dict_from_file(filename):
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

def permutation(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

def preprocess(adata, n_top_genes=3000, get_highly_vars=True):
    if get_highly_vars:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_vars = adata
    else:   
       adata_vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_vars.X, csc_matrix) or isinstance(adata_vars.X, csr_matrix):
       feat = adata_vars.X.toarray()[:, ]
    else:
       feat = adata_vars.X[:, ] 
    
    # data augmentation
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    

def ada_get_cell_type_aware_adj(
        X, 
        adj_0, 
        seed, 
        bi_type, 
        coords, 
        meta_folder, 
        region_col_name,
        n=5, 
        vis=False, 
        eval=False, 
        resolution=None
):
    data_folder = meta_folder
    if resolution != None:
        if osp.exists(osp.join(data_folder, f'pre-cluster_adj_reso={resolution}.pkl')):
            print(
                f">>> Cache exisits: {osp.join(data_folder, f'pre-cluster_adj_reso={resolution}.pkl')}"
            )
            with open(osp.join(data_folder, f'pre-cluster_adj_reso={resolution}.pkl'), 'rb') as file:
                prune_G = pickle.load(file)
        else:
            cell_types = pd.read_csv(osp.join(data_folder, 'cell_types.csv')).iloc[:, 0]
            type_id_list = cell_types[cell_types == bi_type].index.to_list()

            adata = ad.AnnData(X[type_id_list])
            sc.tl.pca(adata, svd_solver='arpack', random_state=seed)
            sc.pp.neighbors(adata, random_state=seed)
            pre_labels = 'expression_louvain_label'

            if vis:
                bi_type_name = bi_type.replace('/', 'or')
                title = f"Pre-cluster_{bi_type_name}"
                fig_name = f"pre-cluster_spatial_{bi_type_name}_reso={resolution}"

                sc.tl.louvain(adata, resolution=resolution, key_added=pre_labels, random_state=seed)
                sc.tl.umap(adata, random_state=seed)
                cluster_labels = [int(x) for x in np.array(adata.obs[pre_labels])]

                label = adata.obs[pre_labels]

                coordinates = coords[type_id_list]
                n_clusters = max(cluster_labels) + 1 # start from 0
                print(f'{n_clusters} clusters in drawing...')
                txt_lines = [ [idx, x] for idx, x in zip(type_id_list, cluster_labels) ]
                np.savetxt(
                    osp.join(data_folder, f"pre-cluster_{bi_type_name}_types_reso={resolution}.txt"), 
                    np.array(txt_lines), 
                    fmt='%3d', 
                    delimiter='\t'
                )

                sc_cluster = plt.scatter(
                    x=coordinates[:, 0], 
                    y=-coordinates[:, 1], 
                    s=5, 
                    c=cluster_labels, 
                    cmap='rainbow'
                )
                if n_clusters <= 3:
                    plt.legend(
                        *sc_cluster.legend_elements(), 
                        bbox_to_anchor=(1, 0.5), 
                        loc='center left', 
                        prop={'size': 9}
                    )
                else:
                    plt.legend(
                        *sc_cluster.legend_elements(num=n_clusters), 
                        bbox_to_anchor=(1, 0.5), 
                        loc='center left', 
                        prop={'size': 9}
                    )
                plt.xticks([])
                plt.yticks([])
                plt.axis('scaled')
                plt.title(f'{title}')
                plt.savefig(osp.join(data_folder, f'{fig_name}.png'), dpi=400, bbox_inches='tight') 
                plt.clf()

            if eval:
                labels_true_full = np.array(
                    pd.read_csv(osp.join(data_folder, 'regions.csv'))[region_col_name].tolist()
                )
                labels_true = labels_true_full[type_id_list]
                labels_pred = [label for idx, label in txt_lines]
                eva(
                    labels_true, 
                    labels_pred, 
                    out_file=osp.join(data_folder, f'pre-cluster_reso={resolution}.txt')
                )

            print('>>> Pruning the graph...')
            if not sp.isspmatrix_coo(adj_0):
                adj = adj_0.tocoo()
            else:
                adj = adj_0.copy()
            adj = adj.astype(np.float32)
            Graph_df = pd.DataFrame({'St': adj.row, 'Ed': adj.col, 'Data': adj.data})
            # print('>>> %d edges before pruning.' %Graph_df.shape[0])
            # pro_labels_dict = dict(zip(list(map(int, label.index)), label)) # FIXME: change the 0 -> 0-th's idx, dict(zip(type_id_list, label))
            pro_labels_dict = dict(zip(type_id_list, label))
            for i in range(X.shape[0]):
                if pro_labels_dict.get(i) is None:
                    pro_labels_dict[i] = '-1'
            Graph_df['St_label'] = Graph_df['St'].map(pro_labels_dict)
            Graph_df['Ed_label'] = Graph_df['Ed'].map(pro_labels_dict)
            # Graph_df = Graph_df.loc[(Graph_df['St_label']==Graph_df['Ed_label']) | (Graph_df['St_label'] == '-1') | (Graph_df['Ed_label'] == '-1')] # Leave the edge connecting not bi_type spot
            Graph_df = Graph_df.loc[(Graph_df['St_label']==Graph_df['Ed_label'])] # Leave only the edge in bi_type groups
            # print('>>> %d edges after pruning.' %Graph_df.shape[0])

            prune_G = sp.coo_matrix((np.ones(Graph_df.shape[0]), (Graph_df['St'], Graph_df['Ed'])))
            with open(osp.join(data_folder, f'pre-cluster_adj_reso={resolution}.pkl'), 'wb') as file:
                pickle.dump(prune_G, file)
    else:
        if osp.exists(osp.join(data_folder, f'pre-cluster_adj_{n}.pkl')):
            print(f">>> Cache exisits: {osp.join(data_folder, f'pre_cluster_adj_{n}.pkl')}")
            with open(osp.join(data_folder, f'pre-cluster_adj_{n}.pkl'), 'rb') as file:
                prune_G = pickle.load(file)
        else:
            cell_types = pd.read_csv(osp.join(data_folder, 'cell_types.csv')).iloc[:, 0]
            type_id_list = cell_types[cell_types == bi_type].index.to_list() # Tid: type_id_list is used for slice, in other word, it contains the idx mapping:
            # For example, 0-th's idx -> 0, 1-th's idx -> 1, 2-th's idx -> 2
            # If we want to do the reverse mapping, just say, 0 -> 0-th's idx, 1 -> 1-th's idx, 2 -> 2-th's idx
            # What should we do, it is use slice operations in the equantion left
            # It's import to notion that the graph after pruning is created from the Graph_df
            # Graph_df, which is the pd.DataFrame, contains three columns: edge start spot idx, edge end spot idx and edge's value
            # How we modify the Graph_df?:
            # First: construct a label dict to map, the dict is created by the scheme: {0: 0 label}
            # Second: Set the new column for the pandas.DataFrame, both edge start spot idx and edge end sport idx
            # Third: Judge from the relation between edge start & end spot idx

            adata = ad.AnnData(X[type_id_list])
            sc.tl.pca(adata, svd_solver='arpack', random_state=seed)
            sc.pp.neighbors(adata, random_state=seed)
            pre_labels = 'expression_louvain_label'
            increment = 0.01
            ret_type_cnt = 0

            # Iterating using the Progress class in rich
            with Progress() as prog:
                reso_lst_sorted = sorted(list(np.arange(increment, 1000, increment)))
                task = prog.add_task('>>> Getting type-aware Adj...', total=len(reso_lst_sorted))
                for reso in reso_lst_sorted:
                    sc.tl.louvain(adata, resolution=reso, key_added=pre_labels, random_state=seed)
                    ret_type_cnt = len(
                        pd.DataFrame(adata.obs[pre_labels]).expression_louvain_label.unique()
                    )
                    # prog.console.print(f"ret_type_cnt={ret_type_cnt}")
                    if ret_type_cnt == n:
                        label = adata.obs[pre_labels]
                        break
                    prog.advance(task)
                print('Resolution:', reso)

            if vis:
                eval_reso = reso
                bi_type_name = bi_type.replace('/', 'or')
                title = f"Pre-cluster_{bi_type_name}"
                fig_name = f"pre-cluster_spatial_{bi_type_name}_{n}"

                sc.tl.louvain(adata, resolution=eval_reso, key_added=pre_labels, random_state=seed)
                sc.tl.umap(adata, random_state=seed)
                cluster_labels = [int(x) for x in np.array(adata.obs[pre_labels])]

                coordinates = coords[type_id_list]
                n_clusters = max(cluster_labels) + 1 # start from 0
                print(f'{n_clusters} clusters in drawing...')
                txt_lines = [ [idx, x] for idx, x in zip(type_id_list, cluster_labels) ]
                np.savetxt(
                    osp.join(data_folder, f"pre-cluster_{bi_type_name}_types_{n}.txt"), 
                    np.array(txt_lines), 
                    fmt='%3d', 
                    delimiter='\t'
                )

                sc_cluster = plt.scatter(
                    x=coordinates[:, 0], 
                    y=-coordinates[:, 1], 
                    s=5, 
                    c=cluster_labels, 
                    cmap='rainbow'
                )
                if n_clusters <= 3:
                    plt.legend(
                        *sc_cluster.legend_elements(), 
                        bbox_to_anchor=(1, 0.5), 
                        loc='center left', 
                        prop={'size': 9}
                    )
                else:
                    plt.legend(
                        *sc_cluster.legend_elements(num=n_clusters), 
                        bbox_to_anchor=(1, 0.5), 
                        loc='center left', 
                        prop={'size': 9}
                    )
                plt.xticks([])
                plt.yticks([])
                plt.axis('scaled')
                plt.title(f'{title}')
                plt.savefig(osp.join(data_folder, f'{fig_name}.png'), dpi=400, bbox_inches='tight') 
                plt.clf()

            if eval:
                labels_true_full = np.array(
                    pd.read_csv(osp.join(data_folder, 'regions.csv'))[region_col_name].tolist()
                )
                labels_true = labels_true_full[type_id_list]
                labels_pred = [label for idx, label in txt_lines]
                eva(labels_true, labels_pred, out_file=osp.join(data_folder, f'pre-cluster_{n}.txt'))

            print('>>> Pruning the graph...')
            if not sp.isspmatrix_coo(adj_0):
                adj = adj_0.tocoo()
            else:
                adj = adj_0.copy()
            adj = adj.astype(np.float32)
            Graph_df = pd.DataFrame({'St': adj.row, 'Ed': adj.col, 'Data': adj.data})
            # print('%d edges before pruning.' %Graph_df.shape[0])
            # pro_labels_dict = dict(zip(list(map(int, label.index)), label)) # FIXME: change the 0 -> 0-th's idx, dict(zip(type_id_list, label))
            pro_labels_dict = dict(zip(type_id_list, label))
            for i in range(X.shape[0]):
                if pro_labels_dict.get(i) is None:
                    pro_labels_dict[i] = '-1'
            Graph_df['St_label'] = Graph_df['St'].map(pro_labels_dict)
            Graph_df['Ed_label'] = Graph_df['Ed'].map(pro_labels_dict)
            # Graph_df = Graph_df.loc[(Graph_df['St_label']==Graph_df['Ed_label']) | (Graph_df['St_label'] == '-1') | (Graph_df['Ed_label'] == '-1')] # Leave the edge connecting not bi_type spot
            Graph_df = Graph_df.loc[(Graph_df['St_label']==Graph_df['Ed_label'])] # Leave only the edge in bi_type groups
            # print('%d edges after pruning.' %Graph_df.shape[0])

            prune_G = sp.coo_matrix((np.ones(Graph_df.shape[0]), (Graph_df['St'], Graph_df['Ed'])))
            with open(osp.join(data_folder, f'pre-cluster_adj_{n}.pkl'), 'wb') as file:
                pickle.dump(prune_G, file)
    return prune_G

def get_bi_type_related_adj(adj_0, y):
    adj_0 = _confirm_coo_mat(adj_0)
    y = _confirm_np_array(y)

    # Use COO format
    new_row = []
    new_col = []
    new_data = []
    for st, ed, e_val in track(zip(adj_0.row, adj_0.col, adj_0.data), description='>>> Selecting...'):
        if y[st] == 1 or y[ed] == 1:
            new_row.append(st)
            new_col.append(ed)
            new_data.append(e_val)

    return sp.coo_matrix((new_data, (new_row, new_col)), shape=adj_0.shape)
