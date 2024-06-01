import os
import time
import torch
import numpy as  np
import os.path as osp
import networkx as nx
import torch_geometric
import torch_geometric.transforms as T
from rich import print
from torch.optim import Adam
from .utils import corruption
from .model import MyDGI, STAEncoder
from torch_geometric.data import Data


def train_model(args, data, adj, adj_prime, coords, y, attn_LRs, device, std_LRs, alpha, 
                sp_re_weight=3):
    model = MyDGI(
        hidden_channels=args.h,
        encoder=STAEncoder(
            n_input=args.n_input, 
            h=args.h, 
            attn_LRs=attn_LRs, 
            init=args.init, 
            std_LRs=std_LRs
        ),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    G = nx.from_numpy_array(adj.cpu().numpy()).to_undirected()
    edge_index = (torch_geometric.utils.convert.from_networkx(G).to(device)).edge_index
    data_obj = Data(edge_index=edge_index, x=data) 
    data_obj.num_nodes = data.shape[0] 
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
    transform = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, 
                                  add_negative_train_samples=False, split_labels=True)
    train_data, _ , _ = transform(data_obj)

    G_prime = nx.from_numpy_array(adj_prime.cpu().numpy()).to_undirected()
    edge_index_prime = (torch_geometric.utils.convert.from_networkx(G_prime).to(device)).edge_index
    data_obj_prime = Data(edge_index=edge_index_prime, x=data) 
    data_obj_prime.num_nodes = data.shape[0] 
    data_obj_prime.train_mask = data_obj_prime.val_mask = data_obj_prime.test_mask = data_obj_prime.y = None
    transform_prime = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, 
                                        add_negative_train_samples=False, split_labels=True)
    train_data_prime, _ , _ = transform_prime(data_obj_prime)

    coords = torch.tensor(coords).float().to(device)
    min_loss = np.inf
    best_params = model.state_dict()

    st = time.time()
    model.train()
    for epoch in range(args.num_epoch):
        epoch_loss = 0.0
        z = model.encoder(data, attn_LRs) # data is gene expression

        if alpha != 0:
            penalty_3 = model.encoder.sp_loss(z, train_data.pos_edge_label_index)
            l_sp_ori = sp_re_weight * penalty_3
            penalty_4 = model.encoder.sp_loss(z, train_data_prime.pos_edge_label_index)
            l_sp_type = sp_re_weight * penalty_4
            l_sp = (1 - alpha) * l_sp_ori + alpha * l_sp_type
        else:
            penalty = model.encoder.sp_loss(z, train_data.pos_edge_label_index)
            l_sp = sp_re_weight * penalty

        optimizer.zero_grad()
        l_sp.backward()
        optimizer.step()
        epoch_loss += l_sp.item()

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_params = model.state_dict()

        if (epoch + 1) % 100 == 0:
            # eva(y, pred, f">{epoch + 1}")
            print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch + 1, epoch_loss))
            print(f'time: {time.time() - st:.3f}, l_sp: {l_sp.item():.4f}')
            st = time.time()

    print(f'Min Loss: {min_loss}')

    model.eval()
    model.load_state_dict(best_params)
    h, lr_weight_dict = model.encoder(data, attn_LRs, print_lambda=True)
    embedding_save_filepath = osp.join(args.embedding_data_path, 'spot_embed.npy')
    save_dir = osp.dirname(embedding_save_filepath)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    np.save(embedding_save_filepath, h.cpu().detach().numpy())

    return lr_weight_dict
