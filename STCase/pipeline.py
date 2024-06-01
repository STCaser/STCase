import time
import torch
import random
import os
import copy
import pickle
import numpy as np
import pandas as pd
import os.path as osp
from rich import print
from .trainer import train_model
from .data_handler import generate_data
from .utils import filter_attn_LRs, replace_attn_LRs
from .utils import draw_sub_type_map, save_dict_to_file
from .utils import ada_get_cell_type_aware_adj, get_bi_type_related_adj


def prepare_args(args):
    args.time_stamp = time.strftime("%m%d_%H%M")

    args.raw_path = args.ds_dir
    args.data_path = 'generated/'
    args.model_path = 'model/'
    args.result_path = 'result/'
    args.embedding_data_path = 'embedding/'
    args.data_name = args.ds_name

    args.device = args.gpu if (torch.cuda.is_available() and args.use_gpu) else 'cpu'

    args.seed = 0
    args.lr_cut = "FULL" # int or "FULL"
    args.h = 10
    args.n_input = 3000
    args.num_epoch = 100
    args.learning_rate = 1e-3

    args.use_norm = False
    args.use_whole_gene = False

    return args

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare(args):
    os.chdir(args.root)
    stcase_args = prepare_args(args)
    seed_everything(stcase_args.seed)
    generate_data(stcase_args)
    return stcase_args

def train(stcase_args):
    meta_folder = osp.join(stcase_args.data_path, stcase_args.data_name)
    types_set = np.loadtxt(
        osp.join(meta_folder, 'types_set.txt'), 
        delimiter='\t', 
        dtype=str
    )
    n_nei = stcase_args.n_nei
    coords = np.load(osp.join(meta_folder, 'coordinates.npy'))
    device = torch.device(stcase_args.device)
    with open(
        osp.join(meta_folder, f'lr_cell_weight_{stcase_args.lr_cut}.pkl'), # TODO
        'rb'
    ) as fp:
        attn_LRs = pickle.load(fp)

    print('>>> Model and Training Details')
    print({**vars(stcase_args)})
    for t in types_set:
        if t not in stcase_args.bad_types and t in stcase_args.target_types:
            print(f">>> For {t.replace('/', 'or')} spots")
            args = copy.deepcopy(stcase_args)
            t_label = np.load(
                osp.join(meta_folder, f"cell_type_indeces_{t.replace('/', 'or')}.npy")
            )
            gene_tag = f"hvg={'FULL' if args.use_whole_gene else args.n_input}"
            method = f"{t.replace('/', 'or')}_alpha={args.alpha}_reso={args.reso}_" + \
                f"cut={args.lr_cut}_{gene_tag}_nei={args.n_nei}"
            args.embedding_data_path = osp.join(
                args.embedding_data_path, 
                args.data_name, 
                method, 
                args.time_stamp
            )  # Tid: out
            if not osp.exists(args.embedding_data_path): # Embedding dir.
                os.makedirs(args.embedding_data_path) 

            # Get data
            X = np.load(osp.join(meta_folder, 'features.npy'))
            print(f'X: {X.shape}')

            with open(osp.join(meta_folder, f'adj_{n_nei}_i.pkl'), 'rb') as fp:
                adj_0 = pickle.load(fp)

            adj = get_bi_type_related_adj(adj_0, t_label)
            if args.alpha == 0:
                adj_prime = adj
            else:
                adj_prime = ada_get_cell_type_aware_adj(
                    X, adj, args.seed, t, coords, meta_folder, args.region_col_name,
                    resolution=args.reso, vis=True, eval=False
                )
            print(f"Edges in Adj for {t.replace('/', 'or')}:", len(adj.data))
            print(f"Edges in Adj' for {t.replace('/', 'or')}:", len(adj_prime.data))

            attn_LRs4t, std_LRs = filter_attn_LRs(
                attn_LRs, adj, 
                cut=args.lr_cut, return_std=True
            )
            attn_LRs4t = replace_attn_LRs(
                attn_LRs4t, t_label, 
                norm=args.use_norm
            )
            print(f"Use {len(attn_LRs4t)} LR pairs")
            attn_LRs4t = {
                k: torch.Tensor(v.toarray()).to(device) for k, v in attn_LRs4t.items()
            }

            print(">>> STCase sub-clustering...")
            adj = torch.Tensor(adj.toarray()).to(device)
            adj_prime = torch.Tensor(adj_prime.toarray()).to(device)
            data = torch.Tensor(X).to(device)

            lr_weight_dict = train_model(
                args, data, adj, adj_prime, coords, t_label, attn_LRs4t, 
                device, std_LRs, alpha=args.alpha
            )

            result_path = osp.join(
                args.result_path, args.data_name, args.time_stamp, method
            )
            if not osp.exists(result_path): # Results dir.
                os.makedirs(result_path) 

            save_dict_to_file(
                lr_weight_dict, osp.join(result_path, f'lr_weight_dict.pkl')
            )
            lr_weight_df = pd.DataFrame(
                zip(list(lr_weight_dict.keys()), list(lr_weight_dict.values())), 
                columns=['lr_name', 'weight']
            )
            lr_weight_df['cell_type'] = t
            lr_weight_df.to_csv(osp.join(result_path, f'lr_weight.csv'))

            print(">>> Drawing map")

            node_embed = np.load(osp.join(args.embedding_data_path, 'spot_embed.npy'))

            if args.n_clusters != -1:
                draw_sub_type_map(
                    t, 
                    args.data_name, 
                    types_set, 
                    node_embed, 
                    method, 
                    args.time_stamp, 
                    args.seed, 
                    fixed_clus_count=args.n_clusters, 
                    cluster_with_fix_reso=False, 
                    eval=not args.wo_anno,
                    region_col_name=args.region_col_name
                )
                draw_sub_type_map(
                    t, 
                    args.data_name, 
                    types_set, 
                    node_embed, 
                    method, 
                    args.time_stamp, 
                    args.seed, 
                    fixed_clus_count=args.n_clusters, 
                    cluster_method='mclust',
                    cluster_with_fix_reso=False,
                    eval=not args.wo_anno,
                    region_col_name=args.region_col_name
                )
            else:
                draw_sub_type_map(
                    t, 
                    args.data_name, 
                    types_set, 
                    node_embed, 
                    method, 
                    args.time_stamp, 
                    args.seed, 
                    cluster_with_fix_reso=True, 
                    eval=False, 
                    resolution=args.reso
                )

            print(method)
