import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import InnerProductDecoder, DeepGraphInfomax
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)


EPS = 1e-15


class LRWeightGraphAttentionLayer(nn.Module):
    """
    LR GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, hidden_features, LRs, init, std_LRs, dropout=0.0, alpha=0.2, concat=False): # in_features should be == hidden_features
        super(LRWeightGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.alpha = alpha
        self.concat = concat

        self.LRs = LRs
        self.LR2ids = {key: idx for idx, key in enumerate(LRs)}

        # [setattr(self, f'lambda_{k}', nn.Parameter(torch.ones(size=(1,)))) for k, v in LRs.items()]
        if init == 'std':
            # right_move = 1e5 # eva not good
            right_move = 1e0
            print(f'right_move={right_move}')
            stds = list(std_LRs.values())
            stds_tensor = (torch.tensor(stds) * right_move).to(next(iter(LRs.values())).device)
            self.lambda_LRs = nn.Parameter(stds_tensor).unsqueeze(1)
        elif init == 'one':
            self.lambda_LRs = nn.Parameter(torch.ones(size=(len(LRs), 1)))
        elif init == 'zero':
            self.lambda_LRs = nn.Parameter(torch.zeros(size=(len(LRs), 1)))
        elif init == 'normal':
            tensor = torch.empty(size=(len(LRs), 1))
            nn.init.normal_(tensor)
            self.lambda_LRs = nn.Parameter(tensor)
        elif init == 're_sum':
            ep = 1e-8
            print(f'ep={ep}')
            sum_tensors = torch.stack(list(LRs.values()), dim=0).sum(dim=(-2, -1))
            reciprocal_tensors = torch.reciprocal(sum_tensors + ep)
            self.lambda_LRs = nn.Parameter(reciprocal_tensors).unsqueeze(1)
        elif init == 'sum':
            sum_tensors = torch.stack(list(LRs.values()), dim=0).sum(dim=(-2, -1))
            self.lambda_LRs = nn.Parameter(sum_tensors).unsqueeze(1)
        elif init == 're_n_e':
            ep = 1e-8
            print(f'ep={ep}')
            non_zero_counts = [torch.count_nonzero(mat).float() for mat in LRs.values()]
            non_zero_counts_tensor = torch.tensor(non_zero_counts).to(non_zero_counts[0].device)
            reciprocal_tensors = torch.reciprocal(non_zero_counts_tensor + ep)
            self.lambda_LRs = nn.Parameter(reciprocal_tensors).unsqueeze(1)
        elif init == 'n_e':
            non_zero_counts = [torch.count_nonzero(mat).float() for mat in LRs.values()]
            non_zero_counts_tensor = torch.tensor(non_zero_counts).to(non_zero_counts[0].device)
            self.lambda_LRs = nn.Parameter(non_zero_counts_tensor).unsqueeze(1)

        print(f"Initial LR pairs' lambda {self.lambda_LRs}")

        self.lr_proj = nn.Linear(in_features + hidden_features, out_features, bias=False)
        self.h_prime_norm = nn.BatchNorm1d(out_features) # XXX

    def forward(self, h, attn_LRs):
        h_lr = torch.zeros(h.shape[0], self.hidden_features).to(h.device)
        lambda_LRs = F.softmax(self.lambda_LRs, dim=0)
        for interaction in self.LRs:
            attn_k_LR = attn_LRs[interaction] # This can be csr format or transferred to coo format
            # lambda_k = getattr(self, f'lambda_{interaction}')
            idx = self.LR2ids[interaction]

            tmp = torch.matmul(attn_k_LR.T, h) # FIXME: Reduce the CUDA usage; tensor.to_sparse()+torch.sparse.mm() or torch.index_select()+torch.einsum()
            # tmp = torch.einsum('ri,ih->rh', attn_k_LR.T, h)
            h_lr += tmp * lambda_LRs[idx]

        h_lr = torch.cat((h, h_lr), dim=1) # Cat_EFT
        h_prime = self.lr_proj(h_lr)
        h_prime = self.h_prime_norm(h_prime)
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class MyDGI(DeepGraphInfomax):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (torch.nn.Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """
    def forward(self, *args, **kwargs):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z = self.encoder(*args, **kwargs)
        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        neg_z = self.encoder(*cor)
        summary = self.summary(pos_z, *args, **kwargs)
        return pos_z, neg_z, summary
    
class STAEncoder(nn.Module):
    def __init__(self, n_input, h, init, attn_LRs, std_LRs):
        super(STAEncoder, self).__init__()

        self.LR_dict = attn_LRs
        # GCN for inter information
        self.gnn_z = LRWeightGraphAttentionLayer(n_input, h, n_input, self.LR_dict, init, std_LRs)

        self.decoder = InnerProductDecoder()
    
    def forward(self, x, attn, print_lambda=False):
        # GAT Module
        h = self.gnn_z(x, attn)

        if print_lambda:
            lr_weight_dict = \
                dict(
                    sorted(
                        dict(
                            zip(
                                self.LR_dict.keys(), F.softmax(self.gnn_z.lambda_LRs, dim=0)[:, 0].detach().cpu().numpy()
                                )).items(), 
                        key=lambda x: x[1], 
                        reverse=True))
            return h, lr_weight_dict

        return h
    
    def sp_loss(self, z, pos_edge_index, neg_edge_index=None):
        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()

        return pos_loss
