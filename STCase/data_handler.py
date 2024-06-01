import os
import time
import pickle
import pandas as pd
import scanpy as sc
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from rich import print
from scipy import sparse
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from .utils import preprocess, get_feature, save_dict_to_file


def get_bi_type(cell_types, generated_data_fold):
    types_set = np.loadtxt(
        osp.join(generated_data_fold, 'types_set.txt'), 
        delimiter='\t', dtype=str
    )
    for t in types_set:
        bi_types_idx = (cell_types == t).astype(int).to_numpy()
        np.save(
            osp.join(generated_data_fold, f"cell_type_indeces_{t.replace('/', 'or')}.npy"), 
            np.array(bi_types_idx)
        )

def get_type(cell_types, generated_data_fold):
    types_set = []
    types_idx = []
    for t in cell_types:
        if not t in types_set:
            types_set.append(t) 
        id = types_set.index(t)
        types_idx.append(id)
    
    np.save(osp.join(generated_data_fold, 'cell_type_indeces.npy'), np.array(types_idx))
    np.savetxt(
        osp.join(generated_data_fold, 'types_set.txt'), 
        np.array(types_set), 
        fmt='%s', delimiter='\t'
    )

def get_adj(generated_data_fold, coordinates, boundary, k):
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold) 
    cell_num = len(coordinates)

    if not osp.exists(osp.join(generated_data_fold, 'distance_array.npy')):
        distance_list = []
        print('>>> Calculating distance matrix, it takes a while...')
        st = time.time()

        distance_list = []
        for j in range(cell_num):
            for i in range (cell_num):
                if i != j:
                    distance_list.append(np.linalg.norm(coordinates[j] - coordinates[i]))

        distance_array = np.array(distance_list)
        np.save(osp.join(generated_data_fold, 'distance_array.npy'), distance_array)
        print(f'Total time: {time.time() - st}s.')
    else:
        distance_array = np.load(osp.join(generated_data_fold, 'distance_array.npy'))

    for threshold in [boundary]:
        num_big = np.where(distance_array < threshold)[0].shape[0]
        print(threshold, num_big, str(num_big / (cell_num * 2)))

        distance_matrix = euclidean_distances(coordinates, coordinates)
        distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
        distance_matrix_threshold_W = np.zeros(distance_matrix.shape)
        for i in range(distance_matrix_threshold_I.shape[0]):
            for j in range(distance_matrix_threshold_I.shape[1]):
                if distance_matrix[i, j] <= threshold and distance_matrix[i, j] > 0:
                    distance_matrix_threshold_I[i, j] = 1
                    distance_matrix_threshold_W[i, j] = distance_matrix[i, j]

        ############### get normalized sparse adjacent matrix
        distance_matrix_threshold_W_N = np.float32(distance_matrix_threshold_W)
        distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I)

        distance_matrix_threshold_W_N_crs = sparse.csr_matrix(distance_matrix_threshold_W_N)
        distance_matrix_thredhold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
        with open(osp.join(generated_data_fold, f'adj_{k}_w.pkl'), 'wb') as fp:
            pickle.dump(distance_matrix_threshold_W_N_crs, fp)
        with open(osp.join(generated_data_fold, f'adj_{k}_i.pkl'), 'wb') as fp:
            pickle.dump(distance_matrix_thredhold_I_N_crs, fp)

def draw_region_map(generated_data_fold, coordinates, region_col_name):
    regions = pd.read_csv(osp.join(generated_data_fold, 'regions.csv'))[region_col_name]
    region2idx = {key: idx for idx, key in enumerate(regions.unique())}
    n_regions = len(region2idx)

    sc_region = plt.scatter(
        x=coordinates[:, 0],
        y=coordinates[:, 1],
        s=5,
        c=[region2idx[key] for key in regions],
        cmap='rainbow'
    )
    plt.legend(
        *sc_region.legend_elements(num=n_regions),
        bbox_to_anchor=(1, 0.5),
        loc='center left',
        prop={'size': 9}
    )

    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    plt.title(region_col_name)
    plt.savefig(
        osp.join(generated_data_fold, 'regions.png'), 
        dpi=400,
        bbox_inches='tight'
    )
    plt.clf()

def draw_map(generated_data_fold, coordinates, label_col_name):
    cell_type_indeces = np.load(osp.join(generated_data_fold, 'cell_type_indeces.npy'))
    n_types = max(cell_type_indeces) + 1
    
    types_set = np.loadtxt(
        osp.join(generated_data_fold, 'types_set.txt'), 
        dtype='|S15',  
        delimiter='\t'
    ).tolist()
    for i, tmp in enumerate(types_set):
        types_set[i] = tmp.decode()

    sc_label = plt.scatter(
        x=coordinates[:, 0], 
        y=-coordinates[:, 1], 
        s=5, 
        c=cell_type_indeces, 
        cmap='rainbow'
    )  
    plt.legend(
        *sc_label.legend_elements(num=n_types), 
        bbox_to_anchor=(1,0.5), 
        loc='center left', 
        prop={'size': 9}
    ) 
    
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    plt.title(label_col_name)
    plt.savefig(
        osp.join(generated_data_fold, 'spacial.png'), 
        dpi=400, 
        bbox_inches='tight'
    ) 
    plt.clf()

def draw_bi_map(generated_data_fold, coordinates, label_col_name):
    types_set = np.loadtxt(
        osp.join(generated_data_fold, 'types_set.txt'), 
        delimiter='\t', 
        dtype=str
    )
    for t in types_set:
        cell_type_indeces = np.load(
            osp.join(
                generated_data_fold, 
                f"cell_type_indeces_{t.replace('/', 'or')}.npy"
            )
        )

        sc_label_bi = plt.scatter(
            x=coordinates[:, 0], 
            y=-coordinates[:, 1], 
            s=5, 
            c=cell_type_indeces, 
            cmap='viridis'
        )  
        plt.legend(
            *sc_label_bi.legend_elements(num=2), # 0 or 1
            bbox_to_anchor=(1,0.5),
            loc='center left',
            prop={'size': 9}
        ) 
        
        plt.xticks([])
        plt.yticks([])
        plt.axis('scaled')
        plt.title(f'{label_col_name}_{t}')
        plt.savefig(
            osp.join(generated_data_fold, f"spacial_{t.replace('/', 'or')}.png"), 
            dpi=400, 
            bbox_inches='tight'
        ) 
        plt.clf()

def generate_data(args):
    data_fold = osp.join(args.raw_path, args.data_name)
    generated_data_fold = osp.join(args.data_path, args.data_name)
    if not osp.exists(generated_data_fold):
        os.makedirs(generated_data_fold)
    adata_h5ad = sc.read_h5ad(osp.join(data_fold, f'{args.h5_name}.h5ad'))

    adata = adata_h5ad.copy()
    adata.var = adata.var.drop('highly_variable', axis=1) if 'highly_variable' in adata.var.keys() else adata.var

    get_highly_vars = True if 'highly_variable' not in adata.var.keys() else False
    preprocess(adata, n_top_genes=args.n_input, get_highly_vars=get_highly_vars)

    if 'feat' not in adata.obsm.keys():
        get_feature(adata, args.use_whole_gene)

    coordinates = np.array(adata.obsm['spatial'])
    features = adata.obsm['feat']

    if args.label_col_name not in adata.obs.columns:
        raise ValueError(f"Column '{args.label_col_name}' does not exist in the h5ad data. Please check the label column name.")
    adata.obs[args.label_col_name].to_csv(osp.join(generated_data_fold, 'cell_types.csv'), index=False)

    if args.region_col_name in adata.obs.columns and not args.wo_anno:
        adata.obs[args.region_col_name].to_csv(osp.join(generated_data_fold, 'regions.csv'), index=False)
        draw_region_map(generated_data_fold, coordinates, args.region_col_name)
    elif args.wo_anno:
        print("Choose to sub-cluster without ground-truth.")
    else:
        print("Sub-clustering region ground-truth not found!")

    np.save(osp.join(generated_data_fold, 'features.npy'), features)
    np.save(osp.join(generated_data_fold, 'coordinates.npy'), coordinates)

    cell_num = len(coordinates)
    adj_coo = coordinates
    k = args.n_nei
    for sigma_num in range(100):
        boundary = 1e8
        for node_idx in range(cell_num):
            tmp = adj_coo[node_idx, :].reshape(1, -1)
            distMat = distance.cdist(tmp, adj_coo, 'euclidean')
            res = distMat.argsort()[: k + 1]
            tmpdist = distMat[0, res[0][1: k + 1]]
            boundary = min(
                np.mean(tmpdist) + sigma_num * np.std(tmpdist), 
                boundary
            )

        edge_num = 0
        for node_idx in range(cell_num):
            tmp = adj_coo[node_idx, :].reshape(1, -1)
            distMat = distance.cdist(tmp, adj_coo, 'euclidean')
            res = distMat.argsort()[: k + 1]
            tmpdist = distMat[0, res[0][1: k + 1]]
            for j in np.arange(1, k + 1):
                if distMat[0, res[0][j]] <= boundary:
                    edge_num += 1

        if edge_num / cell_num > (k - 0.2) and edge_num / cell_num < k:
            print('Boundary: ', boundary)
            print('Average number of neighbor: ', edge_num / cell_num)
            break

    cell_types = adata.obs[args.label_col_name]
    get_adj(generated_data_fold, coordinates, boundary, k)
    get_type(cell_types, generated_data_fold)
    get_bi_type(cell_types, generated_data_fold)
    draw_map(generated_data_fold, coordinates, args.label_col_name)
    draw_bi_map(generated_data_fold, coordinates, args.label_col_name)

    if args.lr_cut == 'FULL':
        cut_lr_cell_weight = adata.uns['LR_cell_weight']
    else:
        items = list(adata.uns['LR_cell_weight'].items())[: args.lr_cut]
        cut_lr_cell_weight = dict(items)
    
    save_dict_to_file(
        cut_lr_cell_weight, 
        osp.join(generated_data_fold, f'lr_cell_weight_{args.lr_cut}.pkl')
    )
    np.save(
        osp.join(generated_data_fold, 'highly_var_gene_symbols.npy'), 
        adata.var[adata.var['highly_variable']].index.to_numpy()
    )
    np.save(
        osp.join(generated_data_fold, f'lr_pairs_{args.lr_cut}.npy'),
        list(cut_lr_cell_weight.keys())
    )
