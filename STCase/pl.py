import matplotlib.pyplot as plt
from netgraph import Graph
from pycirclize import Circos
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import MinMaxScaler
from plotnine import *
import seaborn as sns
from matplotlib import cm
from scipy.spatial.distance import cdist
import random
import igraph as ig
from collections import Counter
from kneed import KneeLocator
from scipy.stats import hypergeom

## Tools

def normalize_list(lst, min_range, max_range):
    # 寻找最小值和最大值
    min_value = min(lst)
    max_value = max(lst)

    # 归一化到指定范围
    normalized_list = [(x - min_value) / (max_value - min_value) * (max_range - min_range) + min_range for x in lst]

    return normalized_list


def normalize_dict(dic, min_range, max_range):
    value_list = [dic[i] for i in dic.keys()]
    value_list = normalize_list(value_list, min_range, max_range)
    dic = {key:weight for key,weight in zip(dic.keys(), value_list)}
    return dic


def detect_outliers_z_score(data, threshold=3):
    """
    使用Z-score方法检测outlier
    """
    z_scores = (data - np.mean(data)) / np.std(data)
    return np.where(np.abs(z_scores) > threshold)[0]
def find_radius(distances, n, scope=6):
    ## 
    nearby_dis = []
    for i in range(n):
        sorted_indices = np.argsort(distances[i])
        nearby_indices = sorted_indices[1:scope + 1]  # 第0个是它自己，不需要计算
        nearby_dis.append([distances[i][a] for a in nearby_indices])
    new_nearby_dis = np.concatenate(nearby_dis)
    # 示例
    outliers = detect_outliers_z_score(new_nearby_dis)
    new_nearby_dis = np.delete(new_nearby_dis, outliers)
    print('The radius is: ' + str(np.max(new_nearby_dis)))
    return np.max(new_nearby_dis)

def Modify_matrix(result):
    n = result.shape[0]
    for i in range(result.shape[0]):
        for j in range(n-1):
            result.iloc[j+1+i,i] = -2
        n = n-1
    return(result)

def prepare_plot_distance_graph(adata,scope,shuffle_num=200):
    n = adata.shape[0]
    coords = adata.obsm['spatial']
    distances = cdist(coords, coords)
    radius = find_radius(distances, n, scope)
    
    edge_array = np.zeros((n, n))
    for i in range(n):
        neighbors = np.where(distances[i] < radius)[0]
        neighbors = np.delete(neighbors, np.where(neighbors == i))
        edge_array[i, neighbors] = 1
        
    cell_type_list = adata.obs['cell_type'].tolist()
    cell_type = list(set(cell_type_list))
    ct_n = len(cell_type)
    edge_num_true = np.zeros((ct_n, ct_n))
    for l in range(ct_n):
        l_ct = cell_type[l]
        index_l = np.where(np.array(cell_type_list)==l_ct)
        edge_array_tmp = edge_array[index_l]
        merged_row = np.sum(edge_array_tmp, axis=0)
        for r in range(ct_n):
            r_ct = cell_type[r]
            edge_num_true[l,r] = np.sum(merged_row[np.where(np.array(cell_type_list)==r_ct)])
    for i in range(edge_num_true.shape[0]):
        edge_num_true[i,i] = edge_num_true[i,i]/2
    
    
    distance_pval = np.zeros((ct_n, ct_n))
    for shuffle in range(shuffle_num):
        edge_num_flase = np.zeros((ct_n, ct_n))
        random.shuffle(cell_type_list)
        for l in range(ct_n):
            l_ct = cell_type[l]
            index_l = np.where(np.array(cell_type_list)==l_ct)
            edge_array_tmp = edge_array[index_l]
            merged_row = np.sum(edge_array_tmp, axis=0)
            for r in range(ct_n):
                r_ct = cell_type[r]
                edge_num_flase[l,r] = np.sum(merged_row[np.where(np.array(cell_type_list)==r_ct)])
        for i in range(edge_num_flase.shape[0]):
            edge_num_flase[i,i] = edge_num_flase[i,i]/2
        distance_pval = distance_pval + (edge_num_true > edge_num_flase)
        
    distance_pval = 1 - (distance_pval/shuffle_num)
    distance_pval_log = -np.log(distance_pval)
    # 使用np.isinf()检查数组中的无穷值
    mask = np.isinf(distance_pval_log)
    # 使用np.where()将无穷值替换为指定的值（例如替换为0）
    distance_pval_log[mask] = -np.log(0.05) * 2
    pval = distance_pval_log.reshape(-1,1)
    distance_pval_log_norm = np.array(normalize_list(pval, -1, 1)).reshape(ct_n,ct_n)
    
    matrix = pd.DataFrame(distance_pval_log_norm, index=cell_type, columns=cell_type)
    row_num = matrix.shape[0]
    col_num = matrix.shape[1]
    
    matrix = Modify_matrix(matrix)
    
    return matrix



def prepare_for_LRI_unit(adata, key):
    ct_interaction_dict = adata.uns[key]
    anndf = pd.DataFrame(index=list(ct_interaction_dict.keys()))
    cell_type = adata.uns['cell_type_list']
    cell_type_indx = {}
    for key,indx in zip(cell_type, range(len(cell_type))):
        cell_type_indx[key] = indx
    for out_ct in cell_type:
        out_indx = cell_type_indx[out_ct]
        for in_ct in cell_type:
            in_indx = cell_type_indx[in_ct]
            for keys in ct_interaction_dict.keys():
                anndf.loc[keys, out_ct+' --> '+in_ct] = ct_interaction_dict[keys][out_indx, in_indx]
    return anndf

def prepare_for_LRI(adata):
    df_dict = {}
    key_list = ['LR_celltype_weight','LR_celltype_mean_weight','LR_celltype_edge_num']
    name = ['weight','weight_per','edge_num']
    for key,n in zip(key_list, name):
        df_dict[n] = prepare_for_LRI_unit(adata, key)

    adata.uns['data_for_LRI'] = df_dict


def prepare_for_spatial_radar(adata, LR_name, min_spot=10):
    coords = adata.obsm['spatial']
    weight_matrix_true = adata.uns['LR_cell_weight'][LR_name]
    weight_matrix_true = weight_matrix_true.toarray()
    receptor_weights = np.sum(weight_matrix_true, axis=0)
    ligand_weights = np.sum(weight_matrix_true, axis=1)
    total_weights = np.array([i+j for i,j in zip(receptor_weights,ligand_weights)])
    # 找到权重总和为0的细胞的索引
    zero_weight_cells = np.where(total_weights == 0)[0]
    # 将权重矩阵中的这些细胞删除
    weight_matrix_true_new = np.delete(weight_matrix_true, zero_weight_cells, axis=0)
    weight_matrix_true_new = np.delete(weight_matrix_true_new, zero_weight_cells, axis=1)
    coords = np.delete(coords, zero_weight_cells, axis=0)
    cell_index = adata.obs.index
    cell_index = np.delete(cell_index, zero_weight_cells, axis=0)
    
    # 结果存储
    ## 权重
    adata.obs[LR_name+'r_weight'] = receptor_weights
    adata.obs[LR_name+'l_weight'] = ligand_weights
    


def get_df(adata, LR_name):
    
    
    lr_res = adata.obs.loc[:,[LR_name+'r_weight',LR_name+'l_weight']]
    lr_res['x_coord'] = list(adata.obsm['spatial'][:,0])
    lr_res['y_coord'] = list(adata.obsm['spatial'][:,1])
    max_coord = np.max([max(list(adata.obsm['spatial'][:,0])),max(list(adata.obsm['spatial'][:,1]))])
    # 示例数据集
    x = lr_res['x_coord'] * 20 / max_coord
    y = lr_res['y_coord'] * 20 / max_coord
    #cluster = lr_res[LR_name+'lr_weight_cluster']
    weight1 = lr_res[LR_name+'r_weight']
    weight2 = lr_res[LR_name+'l_weight']
    df = pd.DataFrame({'x': x, 'y': y, 'weight1': weight1, 'weight2': weight2})
    # 比较权重大小，并为每个点设置颜色和大小
    df['color'] = np.where(df['weight1'] > df['weight2'], 'Receiver', 'Sender')
    df['size'] = np.maximum(df['weight1'], df['weight2'])
    return df




def plot_ct_lr_prepare(adata, 
                       level, 
                       area,
                       num_points,
                       cutoff, 
                       background_type,area_scope):
    if type(level)==str:
        if level in list(adata.uns['LR_celltype_weight'].keys()):
            df = get_df(adata, level)
            df = df[~((df['weight1'] == 0) & (df['weight2'] == 0))]
        elif level in list(adata.uns['LR_pathway_celltype_weight'].keys()):
            def add_list(list1,list2):
                return [x+y for x,y in zip(list1,list2)]
            LR_dict = adata.uns['LR_pair_information']
            pathway_lr = [lr for lr in LR_dict.keys() if LR_dict[lr]['pathway'] == level]
            df_dict = {}
            for lr_name in pathway_lr:
                df_dict[lr_name] = get_df(adata, lr_name)

            list_1 = [0 for _ in range(len(adata.obs.index))]
            list_2 = [0 for _ in range(len(adata.obs.index))]
            for key in df_dict.keys():
                list_1 = add_list(list_1,list(df_dict[key]['weight1']))
                list_2 = add_list(list_2,list(df_dict[key]['weight2']))

            df = df_dict[list(df_dict.keys())[0]][['x','y']]
            df['weight1'] = list_1
            df['weight2'] = list_2

            # 比较权重大小，并为每个点设置颜色和大小
            df['color'] = np.where(df['weight1'] > df['weight2'], 'Receiver', 'Sender')
            df['size'] = np.maximum(df['weight1'], df['weight2'])
            df = df[~((df['weight1'] == 0) & (df['weight2'] == 0))]
        else:
            raise ValueError("Invalid parameter value for parameter 'level'. Only LR pair name and pathway name are allowed.")
    if type(level)==list:
        def add_list(list1,list2):
            return [x+y for x,y in zip(list1,list2)]
        df_dict = {}
        for lr_name in level:
            df_dict[lr_name] = get_df(adata, lr_name)

        list_1 = [0 for _ in range(len(adata.obs.index))]
        list_2 = [0 for _ in range(len(adata.obs.index))]
        for key in df_dict.keys():
            list_1 = add_list(list_1,list(df_dict[key]['weight1']))
            list_2 = add_list(list_2,list(df_dict[key]['weight2']))

        df = df_dict[list(df_dict.keys())[0]][['x','y']]
        df['weight1'] = list_1
        df['weight2'] = list_2

        # 比较权重大小，并为每个点设置颜色和大小
        df['color'] = np.where(df['weight1'] > df['weight2'], 'Receiver', 'Sender')
        df['size'] = np.maximum(df['weight1'], df['weight2'])
        df = df[~((df['weight1'] == 0) & (df['weight2'] == 0))]
    
    if area != None:
        if type(area) == str:
            area = [area]
        coord = adata.obsm['spatial']
        distance = cdist(coord, coord)
        ct_index = [idx for idx,ct in enumerate(adata.obs['cell_type']) if ct in area]
        distance = distance[ct_index]
        min_values = np.min(distance, axis=0)
        indexes = np.where(min_values <= area_scope)
        area_list = adata[indexes].obs.index
        df = df.loc[list(set(list(df.index)) & set(area_list))]
        
    
    if background_type == 'cell_type':
        cell_type_coord = adata.obs.loc[:,['cell_type']]
        lr_cluster_all = cell_type_coord['cell_type']
    #elif background_type == 'community':
    #    cell_type_coord = adata.obs.loc[:,[level+'lr_weight_cluster']]
    #    lr_cluster_all = cell_type_coord[level+'lr_weight_cluster']
    else:
        cell_type_coord = adata.obs.loc[:,[background_type]]
        lr_cluster_all = cell_type_coord[background_type]
    cell_type_coord['x_coord'] = list(adata.obsm['spatial'][:,0])
    cell_type_coord['y_coord'] = list(adata.obsm['spatial'][:,1])
    max_coord = np.max([cell_type_coord['x_coord'].max(),cell_type_coord['y_coord'].max()])
    x_all = cell_type_coord['x_coord'] * 20 / max_coord
    y_all = cell_type_coord['y_coord'] * 20 / max_coord
    df_all = pd.DataFrame({'x_all': x_all, 'y_all': y_all, 'lr_cluster_all': lr_cluster_all})
    
    # 数据处理
    x_min, x_max = df_all['x_all'].min(), df_all['x_all'].max()
    y_min, y_max = df_all['y_all'].min(), df_all['y_all'].max()

    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = np.linspace(y_min, y_max, num_points)
    xv, yv = np.meshgrid(x_vals, y_vals)
    df_fill = pd.DataFrame({'x': xv.flatten(), 'y': yv.flatten()})

    def find_nearest_category(row):
        point = np.array([row['x'], row['y']])
        distances = cdist(df_all[['x_all', 'y_all']], [point])
        closest_index = np.argmin(distances)
        if distances[closest_index] <= cutoff:
            return str(df_all['lr_cluster_all'].iloc[closest_index])
        else:
            return 'none'
    df_fill['nearest_category'] = df_fill.apply(find_nearest_category, axis=1)
    
    return df,df_fill

##function
## (1) CCI
## A. Network plot

def plot_network_total(adata, 
                       tp,
                       ct_list=None,
                       cmap='coolwarm',
                       node_label_size=12, 
                       title_size=15,
                       edge_width_range=(0.5,3),
                       node_size_basis='send',
                       node_size_range=(2,4.5),
                       node_label_offset=0.08,
                       fig_size=(16,8), 
                       save_path=None):
    if node_size_basis not in ['send', 'recive']:
        raise ValueError("Invalid parameter value for parameter 'node_size_basis': {node_size_basis}. Only 'send' and 'recive' are allowed.")
    if tp not in ['weight','edge_num','weight_per','count']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num', 'weight_per' and 'count' are allowed.")
    cell_type = adata.uns['cell_type_list']
    if ct_list == None:
        matrix = adata.uns['LR_celltype_aggregate_weight'][tp]
        matrix = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
    else:
        matrix = adata.uns['LR_celltype_aggregate_weight'][tp]
        matrix = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
        matrix = matrix.loc[ct_list,ct_list]
        cell_type = ct_list
    row_num = matrix.shape[0]
    col_num = matrix.shape[1]
    cell_type_colors = {}
    node_colors = {}
    for i,j in zip(adata.obs['cell_type'].cat.categories, adata.uns['cell_type_colors']):
        cell_type_colors[i] = j
    for i in range(len(cell_type)):
        node_colors[i] = cell_type_colors[cell_type[i]]
    # 将DataFrame转换为一维数组
    values = matrix.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    matrix_norm = pd.DataFrame(scaler.fit_transform(values).reshape(row_num, col_num), columns=cell_type)
    graph_data_count = [(i,j,matrix_norm.iloc[i,j]) for i in range(row_num) for j in range(col_num)]
    graph_data_count_raw = [(i,j,matrix.iloc[i,j]) for i in range(row_num) for j in range(col_num)]
    edge_width = {(u, v) : w for (u, v, w) in graph_data_count_raw}
    edge_width = normalize_dict(edge_width, edge_width_range[0], edge_width_range[1])
    
    
    node_labels = {}
    for i,ct in zip(range(row_num),cell_type):
        node_labels[i] = ct

    node_size_out = {}
    for node in node_labels.keys():
        node_size_out[node] = np.sum([w  for (u, v, w) in graph_data_count_raw if u == node])

    node_size_in = {}
    for node in node_labels.keys():
        node_size_in[node] = np.sum([w  for (u, v, w) in graph_data_count_raw if v == node])

    node_size_out = normalize_dict(node_size_out, node_size_range[0], node_size_range[1])
    node_size_in = normalize_dict(node_size_in, node_size_range[0], node_size_range[1])
    
    if node_size_basis == 'send':
        node_size = node_size_out
    else:
        node_size = node_size_in
    
    if tp == 'weight':
        title = 'Aggregate interaction weight network'
    elif tp == 'weight_per':
        title = 'Aggregate interaction mean weight network'
    elif tp == 'edge_num':
        title = 'Aggregate interaction edge number network'
    else:
        title = 'Aggregate interaction LR number network'
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    
    g1 = Graph(
        graph_data_count,
        node_layout='circular',
        node_labels=node_labels,
        node_label_fontdict=dict(size=node_label_size), 
        node_label_offset=node_label_offset, 
        arrows=True, edge_cmap=cmap,
        node_color=node_colors,
        edge_width = edge_width,
        node_size = node_size_out,
        ax=ax
    )
    ax.set_title(title, fontsize=title_size)

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path) 
        plt.show()


def plot_network_pathway(adata, 
                        pathway_name, 
                        tp,
                        ct_list=None,
                        cmap='coolwarm', 
                        edge_width_range=(0.5,3),
                        title_size=15,
                        node_size_range=(2,4,5),
                        node_size_basis='send',
                        node_label_offset=0.08,
                        save_path=None, 
                        node_label_size=12, 
                        fig_size=(16,8)):
    if node_size_basis not in ['send', 'recive']:
        raise ValueError("Invalid parameter value for parameter 'node_size_basis': {node_size_basis}. Only 'send' and 'recive' are allowed.")
    if tp not in ['weight','edge_num','weight_per','count']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num', 'weight_per' and 'count' are allowed.")
    
    cell_type = adata.uns['cell_type_list']
    if ct_list == None:
        if tp == 'weight':
            matrix = adata.uns['LR_pathway_celltype_weight'][pathway_name]
            title = pathway_name+' interaction weight network'
        elif tp == 'edge_num':
            matrix = adata.uns['LR_pathway_celltype_edge_num'][pathway_name]
            title = pathway_name+' interaction edge number network'
        elif tp == 'weight_per':
            matrix = adata.uns['LR_pathway_celltype_mean_weight'][pathway_name]
            title = pathway_name+' interaction mean weight network'
        else:
            matrix = adata.uns['LR_pathway_celltype_count'][pathway_name]
            title = pathway_name+' interaction LR number network'
        matrix = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
    else:
        if tp == 'weight':
            matrix = adata.uns['LR_pathway_celltype_weight'][pathway_name]
            title = pathway_name+' interaction weight network'
        elif tp == 'edge_num':
            matrix = adata.uns['LR_pathway_celltype_edge_num'][pathway_name]
            title = pathway_name+' interaction edge number network'
        elif tp == 'weight_per':
            matrix = adata.uns['LR_pathway_celltype_mean_weight'][pathway_name]
            title = pathway_name+' interaction mean weight network'
        else:
            matrix = adata.uns['LR_pathway_celltype_count'][pathway_name]
            title = pathway_name+' interaction LR number network'

        matrix = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
        matrix = matrix.loc[ct_list,ct_list]
        cell_type = ct_list


    row_num = matrix.shape[0]
    col_num = matrix.shape[1]
    cell_type_colors = {}
    node_colors = {}
    for i,j in zip(adata.obs['cell_type'].cat.categories, adata.uns['cell_type_colors']):
        cell_type_colors[i] = j
    for i in range(len(cell_type)):
        node_colors[i] = cell_type_colors[cell_type[i]]
    # 将DataFrame转换为一维数组
    values = matrix.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    matrix_norm = pd.DataFrame(scaler.fit_transform(values).reshape(row_num, col_num), columns=cell_type)
    graph_data_count = [(i,j,matrix_norm.iloc[i,j]) for i in range(row_num) for j in range(col_num)]
    graph_data_count_raw = [(i,j,matrix.iloc[i,j]) for i in range(row_num) for j in range(col_num)]
    edge_width = {(u, v) : w for (u, v, w) in graph_data_count_raw}

    edge_width = normalize_dict(edge_width, edge_width_range[0], edge_width_range[1])
    
    
    
    node_labels = {}
    for i,ct in zip(range(row_num),cell_type):
        node_labels[i] = ct

    node_size_out = {}
    for node in node_labels.keys():
        node_size_out[node] = np.sum([w  for (u, v, w) in graph_data_count_raw if u == node])

    node_size_in = {}
    for node in node_labels.keys():
        node_size_in[node] = np.sum([w  for (u, v, w) in graph_data_count_raw if v == node])

    node_size_out = normalize_dict(node_size_out, node_size_range[0], node_size_range[1])
    node_size_in = normalize_dict(node_size_in, node_size_range[0], node_size_range[1])
    
    
    if node_size_basis == 'send':
        node_size = node_size_out
    else:
        node_size = node_size_in
    
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    
    g1 = Graph(
        graph_data_count,
        node_layout='circular',
        node_labels=node_labels,
        node_label_fontdict=dict(size=node_label_size), 
        node_label_offset=node_label_offset, 
        arrows=True, edge_cmap=cmap,
        node_color=node_colors,
        edge_width = edge_width,
        node_size = node_size,
        ax=ax
    )
    ax.set_title(title,fontsize=title_size)

    
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path) 
        plt.show()


def plot_network_single(adata, 
                        LR_name, 
                        tp,
                        ct_list=None,
                        cmap='coolwarm', 
                        edge_width_range=(0.5,3),
                        title_size=15,
                        node_size_range=(2,4,5),
                        node_size_basis='send',
                        node_label_offset=0.08,
                        save_path=None, 
                        node_label_size=12, 
                        fig_size=(16,8)):
    if node_size_basis not in ['send', 'recive']:
        raise ValueError("Invalid parameter value for parameter 'node_size_basis': {node_size_basis}. Only 'send' and 'recive' are allowed.")
    if tp not in ['weight','edge_num','weight_per']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num' and  'weight_per' are allowed.")
    
    cell_type = adata.uns['cell_type_list']
    if ct_list == None:
        if tp == 'weight':
            matrix = adata.uns['LR_celltype_weight'][LR_name]
            title = LR_name+' interaction weight network'
        elif tp == 'edge_num':
            matrix = adata.uns['LR_celltype_edge_num'][LR_name]
            title = LR_name+' interaction edge number network'
        else:
            matrix = adata.uns['LR_celltype_mean_weight'][LR_name]
            title = LR_name+' interaction mean weight network'

        matrix = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
    else:
        if tp == 'weight':
            matrix = adata.uns['LR_celltype_weight'][LR_name]
            title = LR_name+' interaction weight network'
        elif tp == 'edge_num':
            matrix = adata.uns['LR_celltype_edge_num'][LR_name]
            title = LR_name+' interaction edge number network'
        else:
            matrix = adata.uns['LR_celltype_mean_weight'][LR_name]
            title = LR_name+' interaction mean weight network'
        matrix = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
        matrix = matrix.loc[ct_list,ct_list]
        cell_type = ct_list



    row_num = matrix.shape[0]
    col_num = matrix.shape[1]
    cell_type_colors = {}
    node_colors = {}
    for i,j in zip(adata.obs['cell_type'].cat.categories, adata.uns['cell_type_colors']):
        cell_type_colors[i] = j
    for i in range(len(cell_type)):
        node_colors[i] = cell_type_colors[cell_type[i]]
    # 将DataFrame转换为一维数组
    values = matrix.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    matrix_norm = pd.DataFrame(scaler.fit_transform(values).reshape(row_num, col_num), columns=cell_type)
    graph_data_count = [(i,j,matrix_norm.iloc[i,j]) for i in range(row_num) for j in range(col_num)]
    graph_data_count_raw = [(i,j,matrix.iloc[i,j]) for i in range(row_num) for j in range(col_num)]
    edge_width = {(u, v) : w for (u, v, w) in graph_data_count_raw}

    edge_width = normalize_dict(edge_width, edge_width_range[0], edge_width_range[1])
    
    
    
    node_labels = {}
    for i,ct in zip(range(row_num),cell_type):
        node_labels[i] = ct

    node_size_out = {}
    for node in node_labels.keys():
        node_size_out[node] = np.sum([w  for (u, v, w) in graph_data_count_raw if u == node])

    node_size_in = {}
    for node in node_labels.keys():
        node_size_in[node] = np.sum([w  for (u, v, w) in graph_data_count_raw if v == node])

    node_size_out = normalize_dict(node_size_out, node_size_range[0], node_size_range[1])
    node_size_in = normalize_dict(node_size_in, node_size_range[0], node_size_range[1])
    
    
    if node_size_basis == 'send':
        node_size = node_size_out
    else:
        node_size = node_size_in
    
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    
    g1 = Graph(
        graph_data_count,
        node_layout='circular',
        node_labels=node_labels,
        node_label_fontdict=dict(size=node_label_size), 
        node_label_offset=node_label_offset, 
        arrows=True, edge_cmap=cmap,
        node_color=node_colors,
        edge_width = edge_width,
        node_size = node_size,
        ax=ax
    )
    ax.set_title(title,fontsize=title_size)

    
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path) 
        plt.show()



def plot_CCI_network(adata, 
                     level, 
                     tp='weight',
                     ct_list=None,
                     cmap='coolwarm', 
                     edge_width_range=(0.5,3),
                     node_size_range=(2,4,5),
                     node_size_basis='send',
                     node_label_offset=0.08,
                     save_path=None, 
                     node_label_size=12, 
                     fig_size=(16,8)):
    if level=='all':
        plot_network_total(adata, tp=tp, ct_list=ct_list,
                           cmap=cmap,
                           node_label_size=node_label_size, 
                           edge_width_range=edge_width_range,
                           node_size_basis=node_size_basis,
                           node_size_range=node_size_range,
                           node_label_offset=node_label_offset,
                           fig_size=fig_size, 
                           save_path=save_path)
    elif level in list(adata.uns['LR_celltype_weight'].keys()):
        plot_network_single(adata, tp=tp, ct_list=ct_list,
                            LR_name=level, 
                            cmap=cmap,
                            node_label_size=node_label_size, 
                            edge_width_range=edge_width_range,
                            node_size_basis=node_size_basis,
                            node_size_range=node_size_range,
                            node_label_offset=node_label_offset,
                            fig_size=fig_size, 
                            save_path=save_path)
    
    elif level in list(adata.uns['LR_pathway_celltype_weight'].keys()):
        plot_network_pathway(adata, tp=tp, ct_list=ct_list,
                             pathway_name=level, 
                             cmap=cmap,
                             node_label_size=node_label_size, 
                             edge_width_range=edge_width_range,
                             node_size_basis=node_size_basis,
                             node_size_range=node_size_range,
                             node_label_offset=node_label_offset,
                             fig_size=fig_size, 
                             save_path=save_path)
    else:
        raise ValueError("Invalid parameter value for parameter 'level'. Only LR pair name, pathway name and 'all' are allowed.")
    

## B. Chord plot
def plot_chord_total(adata,
                     tp,
                     ct_list=None,
                     label_color='white',
                     ticks_interval=500, 
                     label_size=12,
                     space=3,
                     start=-265,
                     end=95,
                     label_r=93,
                     title_size = 20,
                     save_path=None):
    if tp not in ['weight','edge_num','weight_per','count']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num', 'weight_per' and 'count' are allowed.")
    matrix = adata.uns['LR_celltype_aggregate_weight'][tp]
    cell_type = adata.uns['cell_type_list']
    
    if ct_list == None:
        df = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
    else:
        df = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
        df = df.loc[ct_list,ct_list]
        cell_type = ct_list
    
    node_colors = {}
    for i,j in zip(adata.obs['cell_type'].cat.categories, adata.uns['cell_type_colors']):
        node_colors[i] = j
    
    # 保留小数点后5位
    df = df.round(3)
    row_sums = df.sum(axis=1)
    col_sums = df.sum(axis=0)

    rows_to_drop = row_sums[row_sums == 0].index
    cols_to_drop = col_sums[col_sums == 0].index
    
    drop = list(set(rows_to_drop) & set(cols_to_drop))
    df.drop(index=drop, columns=drop, inplace=True)
    
    circos = Circos.initialize_from_matrix(
        df,
        start=start,
        end=end,
        space=space,
        r_lim=(93, 100),
        cmap=node_colors,
        label_kws=dict(r=label_r, size=label_size, color=label_color),
        ticks_interval=ticks_interval,
        link_kws=dict(ec='black', lw=0.5),
    )
    
    if tp == 'weight':
        title = 'Aggregate interaction weight chord plot'
    elif tp == 'weight_per':
        title = 'Aggregate interaction mean weight chord plot'
    elif tp == 'edge_num':
        title = 'Aggregate interaction edge number chord plot'
    else:
        title = 'Aggregate interaction LR number chord plot'
    
    if save_path == None:
        fig = circos.plotfig()
        plt.title(title, fontsize=title_size)
        plt.show()
    else:
        fig = circos.plotfig()
        plt.title(title, fontsize=title_size)
        fig = circos.savefig(save_path)
        plt.show()

def plot_chord_pathway(adata, 
                       pathway_name,
                       tp,
                       ct_list=None,
                       label_color='white',
                       title_size=15,
                       ticks_interval=10, 
                       label_size=12,
                       space=3,
                       start=-265,
                       end=95,
                       label_r=93,
                       save_path=None):
    if tp not in ['weight','edge_num','weight_per','count']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num', 'weight_per' and 'count' are allowed.")
    
    if tp == 'weight':
        matrix = adata.uns['LR_pathway_celltype_weight'][pathway_name]
        title = pathway_name+' interaction weight chord plot'
    elif tp == 'edge_num':
        matrix = adata.uns['LR_pathway_celltype_edge_num'][pathway_name]
        title = pathway_name+' interaction edge number chord plot'
    elif tp == 'weight_per':
        matrix = adata.uns['LR_pathway_celltype_mean_weight'][pathway_name]
        title = pathway_name+' interaction mean weight chord plot'
    else:
        matrix = adata.uns['LR_pathway_celltype_count'][pathway_name]
        title = pathway_name+' interaction LR number chord plot'
        
    cell_type = adata.uns['cell_type_list']
    if ct_list == None:
        df = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
    else:
        df = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
        df = df.loc[ct_list,ct_list]
        cell_type = ct_list

    node_colors = {}
    for i,j in zip(adata.obs['cell_type'].cat.categories, adata.uns['cell_type_colors']):
        node_colors[i] = j
    
    # 保留小数点后5位
    df = df.round(3)
    row_sums = df.sum(axis=1)
    col_sums = df.sum(axis=0)

    rows_to_drop = row_sums[row_sums == 0].index
    cols_to_drop = col_sums[col_sums == 0].index
    
    drop = list(set(rows_to_drop) & set(cols_to_drop))
    df.drop(index=drop, columns=drop, inplace=True)
    circos = Circos.initialize_from_matrix(
        df,
        start=start,
        end=end,
        space=space,
        r_lim=(93, 100),
        cmap=node_colors,
        label_kws=dict(r=label_r, size=label_size, color=label_color),
        ticks_interval=ticks_interval,
        link_kws=dict(ec="black", lw=0.5),
    )
    
    
    if save_path == None:
        fig = circos.plotfig()
        plt.title(title, fontsize=title_size)
        plt.show()
    else:
        fig = circos.plotfig()
        plt.title(title, fontsize=title_size)
        fig = circos.savefig(save_path)
        plt.show()


def plot_chord_single(adata, 
                      LR_name,
                      tp,
                      ct_list=None,
                      label_color='white',
                      ticks_interval=10, 
                      title_size=15,
                      label_size=12,
                      space=3,
                      start=-265,
                      end=95,
                      label_r=93,
                      save_path=None):
    if tp not in ['weight','edge_num','weight_per']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num' and  'weight_per' are allowed.")
    
    if tp == 'weight':
        matrix = adata.uns['LR_celltype_weight'][LR_name]
        title = LR_name+' interaction weight chord plot'
    elif tp == 'edge_num':
        matrix = adata.uns['LR_celltype_edge_num'][LR_name]
        title = LR_name+' interaction edge number chord plot'
    else:
        matrix = adata.uns['LR_celltype_mean_weight'][LR_name]
        title = LR_name+' interaction mean weight chord plot'
        
    cell_type = adata.uns['cell_type_list']

    if ct_list == None:
        df = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
    else:
        df = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
        df = df.loc[ct_list,ct_list]
        cell_type = ct_list

    node_colors = {}
    for i,j in zip(adata.obs['cell_type'].cat.categories, adata.uns['cell_type_colors']):
        node_colors[i] = j
    
    # 保留小数点后5位
    df = df.round(3)
    row_sums = df.sum(axis=1)
    col_sums = df.sum(axis=0)

    rows_to_drop = row_sums[row_sums == 0].index
    cols_to_drop = col_sums[col_sums == 0].index
    
    drop = list(set(rows_to_drop) & set(cols_to_drop))
    df.drop(index=drop, columns=drop, inplace=True)
    circos = Circos.initialize_from_matrix(
        df,
        start=start,
        end=end,
        space=space,
        r_lim=(93, 100),
        cmap=node_colors,
        label_kws=dict(r=label_r, size=label_size, color=label_color),
        ticks_interval=ticks_interval,
        link_kws=dict(ec="black", lw=0.5),
    )
    
    if save_path == None:
        fig = circos.plotfig()
        plt.title(title,fontsize=title_size)
        plt.show()
    else:
        fig = circos.plotfig()
        plt.title(title,fontsize=title_size)
        fig = circos.savefig(save_path)
        plt.show()



def plot_CCI_chord(adata,
                   level,
                   tp='weight',
                   ct_list=None,
                   label_color='white',
                   ticks_interval=100, 
                   label_size=12,
                   space=3,
                   start=-265,
                   end=95,
                   label_r=93,
                   title_size = 15,
                   save_path=None):

    if level=='all':
        plot_chord_total(adata,tp=tp,ct_list=ct_list,
                         label_color=label_color,
                         ticks_interval=ticks_interval, 
                         label_size=label_size,
                         space=space,
                         start=start,
                         end=end,
                         label_r=label_r,
                         title_size = title_size,
                         save_path=save_path)
    elif level in list(adata.uns['LR_celltype_weight'].keys()):
        plot_chord_single(adata, tp=tp,ct_list=ct_list,
                          LR_name=level,
                          label_color=label_color,
                          ticks_interval=ticks_interval, 
                          title_size=title_size,
                          label_size=label_size,
                          space=space,
                          start=start,
                          end=end,
                          label_r=label_r,
                          save_path=save_path)
    
    elif level in list(adata.uns['LR_pathway_celltype_weight'].keys()):
        plot_chord_pathway(adata, tp=tp,ct_list=ct_list,
                           pathway_name=level,
                           label_color=label_color,
                           title_size=title_size,
                           ticks_interval=ticks_interval, 
                           label_size=label_size,
                           space=space,
                           start=start,
                           end=end,
                           label_r=label_r,
                           save_path=save_path)
    else:
        raise ValueError("Invalid parameter value for parameter 'level'. Only LR pair name, pathway name and 'all' are allowed.")


## C.Heatmap
def plot_heatmap_total(adata, 
                       tp,
                       ct_list=None,
                       cmap='coolwarm',
                       annot=True, 
                       row_cluster=True, 
                       col_cluster=True, 
                       linewidths=1, 
                       vmax=None, 
                       vmin=None,
                       save_path=None,
                       title_size=15):
    if tp not in ['weight','edge_num','weight_per','count']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num', 'weight_per' and 'count' are allowed.")
    
    matrix = adata.uns['LR_celltype_aggregate_weight'][tp]
    cell_type = adata.uns['cell_type_list']
    
    if ct_list == None:
        df = pd.DataFrame(matrix,index=cell_type, columns=cell_type)
    else:
        df = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
        df = df.loc[ct_list,ct_list]
        cell_type = ct_list
        
        
    if tp == 'weight':
        title = 'Aggregate interaction weight heatmap'
    elif tp == 'weight_per':
        title = 'Aggregate interaction mean weight heatmap'
    elif tp == 'edge_num':
        title = 'Aggregate interaction edge number heatmap'
    else:
        title = 'Aggregate interaction LR number heatmap'
    
    fig = sns.clustermap(df, 
                   cmap=cmap, annot=annot, row_cluster=row_cluster, col_cluster=col_cluster, linewidths=linewidths, vmax=vmax, vmin=vmin)
    #add title
    plt.title(title,fontsize=title_size,loc='left')
    if save_path == None:
        pass
    else:
        fig.savefig(save_path, dpi = 400)
    return fig

def plot_heatmap_pathway(adata,
                         pathway_name, 
                         tp,
                         ct_list=None,
                         cmap='RdPu',
                         annot=True, 
                         row_cluster=True, 
                         col_cluster=True, 
                         linewidths=1, 
                         vmax=None, 
                         vmin=None,
                         save_path=None,
                         title_size=15):
    if tp not in ['weight','edge_num','weight_per','count']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num', 'weight_per' and 'count' are allowed.")
    
    if tp == 'weight':
        matrix = adata.uns['LR_pathway_celltype_weight'][pathway_name]
        title = pathway_name+' interaction weight heatmap'
    elif tp == 'edge_num':
        matrix = adata.uns['LR_pathway_celltype_edge_num'][pathway_name]
        title = pathway_name+' interaction edge number heatmap'
    elif tp == 'weight_per':
        matrix = adata.uns['LR_pathway_celltype_mean_weight'][pathway_name]
        title = pathway_name+' interaction mean weight heatmap'
    else:
        matrix = adata.uns['LR_pathway_celltype_count'][pathway_name]
        title = pathway_name+' interaction LR number heatmap'
        
    cell_type = adata.uns['cell_type_list']

    if ct_list == None:
        df = pd.DataFrame(matrix,index=cell_type, columns=cell_type)
    else:
        df = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
        df = df.loc[ct_list,ct_list]
        cell_type = ct_list

    fig = sns.clustermap(df, 
                   cmap=cmap, annot=annot, row_cluster=row_cluster, col_cluster=col_cluster, linewidths=linewidths, vmax=vmax, vmin=vmin)
    #add title
    plt.title(title,fontsize=title_size,loc='left')
    if save_path == None:
        pass
    else:
        fig.savefig(save_path, dpi = 400)
    return fig


def plot_heatmap_list(adata,
                         lr_list, 
                         tp,
                         ct_list=None,
                         cmap='RdPu',
                         annot=True, 
                         row_cluster=True, 
                         col_cluster=True, 
                         linewidths=1, 
                         vmax=None, 
                         vmin=None,
                         save_path=None,
                         title_size=15):
    if tp not in ['weight','edge_num','weight_per']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num' and 'weight_per' are allowed.")
    
    if tp == 'weight':
        LR_pathway_dict = {lr:adata.uns['LR_celltype_weight'][lr] for lr in lr_list}
        matrix = np.zeros_like(next(iter(LR_pathway_dict.values())))
        for array in LR_pathway_dict.values():
            matrix += array
        title = 'Interaction weight heatmap'
    elif tp == 'edge_num':
        LR_pathway_edge_dict = {lr:adata.uns['LR_celltype_edge_num'][lr] for lr in lr_list}
        matrix = np.zeros_like(next(iter(LR_pathway_edge_dict.values())))
        for array in LR_pathway_edge_dict.values():
            matrix += array
        title = 'Interaction edge number heatmap'
    else:
        LR_pathway_mean_dict = {lr:adata.uns['LR_celltype_mean_weight'][lr] for lr in lr_list}
        matrix = np.zeros_like(next(iter(LR_pathway_mean_dict.values())))
        for array in LR_pathway_mean_dict.values():
            matrix += array
        title = 'Interaction mean weight heatmap'
        
    cell_type = adata.uns['cell_type_list']
    if ct_list == None:
        df = pd.DataFrame(matrix,index=cell_type, columns=cell_type)
    else:
        df = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
        df = df.loc[ct_list,ct_list]
        cell_type = ct_list
    fig = sns.clustermap(df, 
                   cmap=cmap, annot=annot, row_cluster=row_cluster, col_cluster=col_cluster, linewidths=linewidths, vmax=vmax, vmin=vmin)
    #add title
    plt.title(title,fontsize=title_size,loc='left')
    if save_path == None:
        pass
    else:
        fig.savefig(save_path, dpi = 400)
    return fig

def plot_heatmap_single(adata,
                        LR_name, 
                        tp,
                        ct_list=None,
                        cmap='RdPu',
                        annot=True, 
                        row_cluster=True, 
                        col_cluster=True, 
                        linewidths=1, 
                        vmax=None, 
                        vmin=None,
                        save_path=None,
                        title_size=15):
    if tp not in ['weight','edge_num','weight_per']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num' and  'weight_per' are allowed.")
    
    if tp == 'weight':
        matrix = adata.uns['LR_celltype_weight'][LR_name]
        title = LR_name+' interaction weight heatmap'
    elif tp == 'edge_num':
        matrix = adata.uns['LR_celltype_edge_num'][LR_name]
        title = LR_name+' interaction edge number heatmap'
    else:
        matrix = adata.uns['LR_celltype_mean_weight'][LR_name]
        title = LR_name+' interaction mean weight heatmap'
    
    cell_type = adata.uns['cell_type_list']
    if ct_list == None:
        df = pd.DataFrame(matrix,index=cell_type, columns=cell_type)
    else:
        df = pd.DataFrame(matrix, index=cell_type, columns=cell_type)
        df = df.loc[ct_list,ct_list]
        cell_type = ct_list



    fig = sns.clustermap(df, 
                   cmap=cmap, annot=annot, row_cluster=row_cluster, col_cluster=col_cluster, linewidths=linewidths, vmax=vmax, vmin=vmin)
    #add title
    plt.title(title,fontsize=title_size, loc='left')
    if save_path == None:
        pass
    else:
        fig.savefig(save_path, dpi = 400)
    return fig


def plot_CCI_heatmap(adata,
                     level, 
                     tp='weight',
                     ct_list=None,
                     cmap='RdPu',
                     annot=True, 
                     row_cluster=True, 
                     col_cluster=True, 
                     linewidths=1, 
                     vmax=None, 
                     vmin=None,
                     save_path=None,
                     title_size=15):

    if level=='all':
        plot_heatmap_total(adata, tp=tp, ct_list=ct_list,
                           cmap=cmap,
                           annot=annot, 
                           row_cluster=row_cluster, 
                           col_cluster=col_cluster, 
                           linewidths=linewidths, 
                           vmax=vmax, 
                           vmin=vmin,
                           save_path=save_path,
                           title_size=title_size)
    elif level in list(adata.uns['LR_pathway_celltype_weight'].keys()):
        plot_heatmap_pathway(adata, tp=tp,ct_list=ct_list,
                             pathway_name=level, 
                             cmap=cmap,
                             annot=annot, 
                             row_cluster=row_cluster, 
                             col_cluster=col_cluster, 
                             linewidths=linewidths, 
                             vmax=vmax, 
                             vmin=vmin,
                             save_path=save_path,
                             title_size=title_size)
    
    elif level in list(adata.uns['LR_celltype_weight'].keys()):
        plot_heatmap_single(adata, tp=tp, ct_list=ct_list,
                            LR_name=level, 
                            cmap=cmap,
                            annot=annot, 
                            row_cluster=row_cluster, 
                            col_cluster=col_cluster, 
                            linewidths=linewidths, 
                            vmax=vmax, 
                            vmin=vmin,
                            save_path=save_path,
                            title_size=title_size)
    
    elif type(level)==list:
        plot_heatmap_list(adata,
                         lr_list=level, 
                         tp=tp,
                         ct_list=ct_list,
                         cmap=cmap,
                        annot=annot, 
                        row_cluster=row_cluster, 
                        col_cluster=col_cluster, 
                        linewidths=linewidths, 
                        vmax=vmax, 
                        vmin=vmin,
                        save_path=save_path,
                        title_size=title_size)

    else:
        raise ValueError("Invalid parameter value for parameter 'level'. Only LR pair name, pathway name and 'all' are allowed.")


## (2) LRI
## A. Natwork plot
def plot_LRI_network(adata, 
                     source, 
                     target, 
                     lr_list=None,
                     tp='weight',
                     top_n=20, 
                     node_size=0.15, 
                     cmap='coolwarm',
                     color={'ligand':'#F79327','receptor':'#8696FE','lr':'#95E1D3'},
                     figsize=(8,8), 
                     save_path=None,
                     title_size=15):
    if tp not in ['weight','edge_num','weight_per']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num' and  'weight_per' are allowed.")
    if  'data_for_LRI' in adata.uns.keys():
        df = adata.uns['data_for_LRI'][tp]
    else:
        prepare_for_LRI(adata)
        df = adata.uns['data_for_LRI'][tp]
    if lr_list==None:
        top_indices = df[source + ' --> ' + target].nlargest(top_n).index
    else:
        top_indices = lr_list
    plot_df = pd.DataFrame(columns=['ligand','receptor'])
    for index in top_indices:
        plot_df.loc[index] = [adata.uns['LR_pair_information'][index]['ligand'], adata.uns['LR_pair_information'][index]['receptor']]
    plot_df['values'] = df.loc[top_indices][[source + ' --> ' + target]]
    
    g = ig.Graph.TupleList(plot_df.itertuples(index=False), directed=False, edge_attrs=['Values'])
    # 设置节点和边的样式
    visual_style = {}
    visual_style['vertex_size'] = node_size  # 节点大小
    # 根据边权重设置边宽度
    edge_weight = [weight for weight in g.es['Values']]
    edge_widths = normalize_list(edge_weight, 1, 6)
    edge_colors = normalize_list(edge_weight, -1, 1)
    visual_style['edge_width'] = edge_widths
    
    cmap = cm.get_cmap(cmap)  # 使用"cool"色图
    edge_colors = [cmap(weight) for weight in edge_colors]
    visual_style['edge_color'] = edge_colors

    # 根据节点属性设置节点颜色
    node_colors = []
    for vertex in g.vs:
        if vertex['name'] in plot_df['ligand'].unique() and vertex['name'] in plot_df['receptor'].unique():
            node_colors.append(color['lr'])  # 既是受体也是配体的节点颜色为紫色
        elif vertex['name'] in plot_df['ligand'].unique():
            node_colors.append(color['ligand'])  # 配体节点颜色为蓝色
        elif vertex['name'] in plot_df['receptor'].unique():
            node_colors.append(color['receptor'])  # 受体节点颜色为红色

    visual_style['vertex_color'] = node_colors

    # 设置节点标签
    vertex_labels = [str(v) for v in g.vs['name']]

    # 将节点围成圆形布局
    layout = g.layout_circle()
    coord_list = layout.coords
    node_df = g.get_vertex_dataframe()

    ligand_index = [index  for index in range(node_df.shape[0]) if node_df.iloc[index,0] in plot_df['ligand'].unique()]
    receptor_index = [index  for index in range(node_df.shape[0]) if node_df.iloc[index,0] in plot_df['receptor'].unique()]
    lr_index = list(set(ligand_index) & set(receptor_index))
    ligand_index = set(ligand_index) - set(lr_index)
    receptor_index = set(receptor_index) - set(lr_index)

    coord_dict = {}
    for i,j in zip(ligand_index, range(len(ligand_index))):
        coord_dict[i] = coord_list[j]
    num = len(ligand_index)
    for i,j in zip(receptor_index, range(len(receptor_index))):
        coord_dict[i] = coord_list[num + j]
    num = num + len(receptor_index)
    for i,j in zip(lr_index, range(len(lr_index))):
        coord_dict[i] = coord_list[num + j]

    coords = []
    for indx in node_df.index:
        coords.append(coord_dict[indx])

    if tp == 'weight':
        title = ' LRI weight network'
    if tp == 'weight_per':
        title = ' LRI mean weight network'
    if tp == 'edge_num':
        title = ' LRI edge number network'
    # 绘制网络图
    if save_path == None:
        fig, ax = plt.subplots(figsize=figsize)
        ig.plot(g, target=ax, layout=coords, vertex_label=vertex_labels,edge_curved=.2,
                vertex_label_size=10,vertex_label_cex=.7,
                **visual_style)
        plt.title(source + ' --> ' + target + title, fontsize=title_size)
        plt.show()
        return
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ig.plot(g, target=ax, layout=coords, vertex_label=vertex_labels,edge_curved=.2,
                vertex_label_size=10,vertex_label_cex=.7,
                **visual_style)
        plt.title(source + ' --> ' + target + title, fontsize=title_size)
        fig.savefig(save_path, dpi=800)
        plt.show()
        return


## B.Chord plot
def plot_LRI_chord(adata, 
                   source, 
                   target,
                   lr_list=None, 
                   tp='weight',
                   top_n=20, 
                   cmap='Set3',
                   color={'ligand':'#F79327','receptor':'#8696FE','lr':'#95E1D3'},
                   figsize=(8,8), 
                   lr_color=False, 
                   save_path=None,
                   title_size=15,
                   title_loc='center',
                   start=-270,
                   end=90,
                   space=2,
                   label_size=12,
                   label_orientation="vertical"):
    if tp not in ['weight','edge_num','weight_per']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num' and  'weight_per' are allowed.")
    
    if  'data_for_LRI' in adata.uns.keys():
        df = adata.uns['data_for_LRI'][tp]
    else:
        prepare_for_LRI(adata)
        df = adata.uns['data_for_LRI'][tp]
    if lr_list==None:
        top_indices = df[source + ' --> ' + target].nlargest(top_n).index
    else:
        top_indices = lr_list
    plot_df = pd.DataFrame(columns=['ligand','receptor'])
    for index in top_indices:
        plot_df.loc[index] = [adata.uns['LR_pair_information'][index]['ligand'], adata.uns['LR_pair_information'][index]['receptor']]
    plot_df['values'] = df.loc[top_indices][[source + ' --> ' + target]]
    # 使用pivot转换DataFrame，并替换NaN值为0
    pivot_df = plot_df.pivot(index='ligand', columns='receptor', values='values').fillna(0)
     
    # 删除行和为0的行
    pivot_df = pivot_df.loc[~(pivot_df.sum(axis=1) == 0)]
    # 删除列和为0的列
    pivot_df = pivot_df.loc[:, ~(pivot_df.sum(axis=0) == 0)]

    if lr_color == True:
        ligand_list = pivot_df.index.tolist()
        receptor_list = pivot_df.columns.tolist()
        lr_list = list(set(ligand_list) & set(receptor_list))
        cmap = {}
        for lig in ligand_list:
            cmap[lig] = color['ligand']
        for rec in receptor_list:
            cmap[rec] = color['receptor']
        for lr in lr_list:
            cmap[lr] = color['lr']
    
    if tp == 'weight':
        title = ' LRI weight network'
    if tp == 'weight_per':
        title = ' LRI mean weight network'
    if tp == 'edge_num':
        title = ' LRI edge number network'
    # Initialize from matrix (Can also directly load tsv matrix file)
    circos = Circos.initialize_from_matrix(
        pivot_df,
        start = start,
        end = end,
        space=space,
        cmap=cmap,
        label_kws=dict(size=label_size,orientation=label_orientation),
        link_kws=dict(direction=1, ec="black", lw=0.5, color='grey')
    )
    if save_path == None:
        fig = circos.plotfig()
        plt.title(source + ' --> ' + target + title,fontsize=title_size,loc=title_loc)
        plt.show()
    else:
        fig = circos.plotfig()
        plt.title(source + ' --> ' + target + title,fontsize=title_size,loc=title_loc)
        fig = circos.savefig(save_path)
        plt.show()


## C.Dotplot
def plot_LRI_dotplot(adata, 
                     source, 
                     target, 
                     lr_list=None,
                     tp='weight',
                     figsize=(6.4,4.8),
                     top_n=5,
                     title_size=15,
                     x_label_size=10,
                     y_label_size=10,
                     spot_size_range=(5,10),
                     save_path=None):
    if tp not in ['weight','edge_num','weight_per']:
        raise ValueError("Invalid parameter value for parameter 'tp': {tp}. Only 'weight', 'edge_num' and  'weight_per' are allowed.")
    
    if type(source)==str:
        source = [source]
    if type(target)==str:
        target = [target]
    
    if  'data_for_LRI' in adata.uns.keys():
        df = adata.uns['data_for_LRI'][tp]
    else:
        prepare_for_LRI(adata)
        df = adata.uns['data_for_LRI'][tp]
    col_name_list = []
    top_indices_list = []
    for s,t in zip(source, target):
        col_name = s + ' --> ' + t
        col_name_list.append(col_name)
        top_indices = df[col_name].nlargest(top_n).index
        top_indices = top_indices.tolist()
        top_indices_list = top_indices_list + top_indices
    if lr_list == None:
        df_tmp = df.loc[top_indices_list][col_name_list]
    else:
        df_tmp = df.loc[lr_list][col_name_list]
    # 转置DataFrame
    df_tmp = df_tmp.T.reset_index().melt(id_vars='index', value_name='Value', var_name='Y')

    if tp == 'weight':
        title = 'LRI weight Dotplot'
    if tp == 'weight_per':
        title = 'LRI mean weight Dotplot'
    if tp == 'edge_num':
        title = 'LRI edge number Dotplot'
    # 绘制dotplot
    plot = (
        ggplot(df_tmp, aes(x='index', y='Y', size='Value', fill='Value')) +
        geom_point(shape='o') +
        scale_size(range=(spot_size_range[0], spot_size_range[1])) +
        scale_fill_gradient(low='lightblue', high='darkblue') +
        labs(x='Cell type', y='LR pair') +
        ggtitle(title) +
        theme_bw() +
        theme(plot_title = element_text(hjust = 0.5, size=title_size),
              axis_title_x = element_text(size=x_label_size),
              axis_title_y = element_text(size=y_label_size),
                 figure_size=figsize)
    )
    if save_path != None:
        plot.save(save_path, height=figsize[1], width=figsize[0])
    return plot


## (3) Community discovery downstream visualization
## A.Spatial
def plot_Community_spatial(adata, 
                           level, 
                           area=None,
                           title=None,
                           num_points=100, 
                           cutoff=0.5, 
                           figure_size=(8,8), 
                           background_type='cell_type', 
                           save_path=None,
                           spot_size=(1,5),
                           title_size=15, area_scope=0):
    
    #if background_type not in ['cell_type', 'community']:
    #    raise ValueError("Invalid parameter value for parameter 'background_type': {background_type}. Only 'cell_type' and 'community' are allowed.")
    if level in list(adata.uns['LR_pathway_celltype_weight'].keys()):
        LR_dict = adata.uns['LR_pair_information']
        pathway_lr = [lr for lr in LR_dict.keys() if LR_dict[lr]['pathway'] == level]
        for lr in pathway_lr:
            prepare_for_spatial_radar(adata, lr , min_spot=0)
    elif level in list(adata.uns['LR_celltype_weight'].keys()):
        prepare_for_spatial_radar(adata, level , min_spot=0)
    else:
        raise ValueError("Invalid parameter value for parameter 'level'. Only LR pair name and pathway name are allowed.")
    df, df_fill = plot_ct_lr_prepare(adata, level, area, num_points,cutoff, background_type=background_type, area_scope=area_scope)
    
    
    if len(adata.uns['cell_type_colors'][0])==9:
         adata.uns['cell_type_colors'] = np.array([i[:7] for i in adata.uns['cell_type_colors']], dtype=object)
    if background_type=='community':
        sc.pl.spatial(adata,color=[level+'lr_weight_cluster', level+'r_weight', level+'l_weight'])
        color_mapping = {i:j for i,j in zip(adata.obs[level+'lr_weight_cluster'].cat.categories, 
                                            adata.uns[level+'lr_weight_cluster_colors'])}
    elif background_type=='cell_type':
        #sc.pl.spatial(adata,color=['cell_type'])
        color_mapping = {i:j for i,j in zip(adata.obs['cell_type'].cat.categories, 
                                            adata.uns['cell_type_colors'])}
    else:
        color_mapping = {i:j for i,j in zip(adata.obs[background_type].cat.categories,
                                            adata.uns[background_type+'_colors'])}
    color_mapping['none'] = 'white'
    
    if title == None:
        title = level
    # 创建一个自定义颜色映射
    color_mapping_point = {'Receiver': 'red', 'Sender': 'blue'}
    plot = (
    ggplot(df_fill,aes(x='x',y='y'))+
    geom_tile(aes(fill = 'nearest_category'), alpha = 0.3)+
    geom_point(df, aes(x = 'x', y = 'y', color = 'color', size = 'size'), alpha = 0.5)+
    scale_fill_manual(values = color_mapping)+
    scale_size(range = (spot_size[0],spot_size[1]))+
    scale_color_manual(values = color_mapping_point)+
    theme_classic()+
    labs(x='spatial1', y='spatlal2', title=title)+
    theme(figure_size=figure_size, axis_line=element_blank(),
              axis_text_x=element_blank(),
              axis_text_y=element_blank(),
              axis_ticks=element_blank(),
              panel_border=element_rect(color="black", size=1),
              plot_title=element_text(hjust=0.5, vjust=0.5,size=title_size))+
    scale_y_reverse()
    )
    if save_path == None:
        return plot
    else:
        plot.save(save_path, height=figure_size[0], width=figure_size[1], dpi=600)
        return plot

## (4) Others
## A.Distance Network plot
def plot_distance_network(adata,
                          scope=6,
                          title_size=15,
                          shuffle_num=200,
                          cmap='coolwarm',
                          node_label_size=12,
                          node_label_offset=0.08,
                          edge_width=2,
                          node_size=4,
                          figsize=(8,8),
                          ct_list=None,
                          save_path=None):
    
    matrix = prepare_plot_distance_graph(adata, scope=scope, shuffle_num=shuffle_num)
    if ct_list == None:
        cell_type = matrix.index.tolist()
    else:
        matrix = matrix.loc[ct_list,ct_list]
        cell_type = ct_list
    
    graph_data_count = [(i,j,matrix.loc[i,j]) for i in cell_type for j in cell_type]
    graph_data_count = [(u,v,z) for (u,v,z) in graph_data_count if z !=-2]
    
    cell_type_colors = {}
    for i,j in zip(adata.obs['cell_type'].cat.categories, adata.uns['cell_type_colors']):
        cell_type_colors[i] = j
    
    fig, ax = plt.subplots(figsize=figsize)
    g1 = Graph(
            graph_data_count,
            node_layout='circular',
            node_labels=True,
            node_label_fontdict=dict(size=node_label_size), node_label_offset=node_label_offset, 
            edge_cmap=cmap,
            node_color = cell_type_colors,
            edge_width = edge_width,
            node_size = node_size,
            ax = ax
        )
    ax.set_title('Cell type distance network',fontsize=title_size)
    
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path) 
        plt.show()



## B.Spatial plot for two interaction cell type

def plot_spatial(adata,
                 cell_type1,
                 cell_type2,
                 save_path = None):
    ## 绘制某两个细胞类型相互作用的区域
    obs_meta = adata.obs.copy()
    n = adata.obs.shape[0]
    obs_meta.index = range(n)
    
    ct_index = obs_meta[obs_meta['cell_type'].isin([cell_type1, cell_type2])].index
    ct1_index = obs_meta[obs_meta['cell_type'].isin([cell_type1])].index
    ct2_index = obs_meta[obs_meta['cell_type'].isin([cell_type2])].index
    non_ct_index = list(set(list(obs_meta.index)) - set(list(ct_index)))
    
    LR_cell_weight = adata.uns['LR_cell_weight']
    cw = np.zeros((n, n))
    for key in LR_cell_weight.keys():
        cw = cw + LR_cell_weight[key].toarray()
    
    ## 其他无关的细胞类型对应的行和列赋值为0
    cw[non_ct_index,:] = 0
    cw[:,non_ct_index] = 0
    mask = np.zeros_like(cw, dtype=bool)
    ## 自分泌的赋值为0
    # 将指定行和列的布尔值设置为True
    for idx in ct1_index:
        mask[ct1_index,idx] = True
    for idx in ct2_index:
        mask[ct2_index,idx] = True
    # 将布尔数组中对应位置为True的元素设置为0
    cw[mask] = 0
    
    index_keep = []
    for i in range(n):
        score_cell = np.sum(cw[:, i]) + np.sum(cw[i, :])
        if score_cell != 0:
            index_keep.append(i)
            
    adata_sub = adata[index_keep,]
    sc.pl.spatial(adata_sub, color='cell_type',save=save_path)

## C. pathway enrichment

def pathway_enrichment_plot(pathway_enrichment_df):
    pathway_enrichment_df = pathway_enrichment_df.sort_values(by='logpval', ascending=True)
    cmap = plt.get_cmap('Blues')
    # 计算每个柱子对应的颜色值
    colors = [cmap(i / len(pathway_enrichment_df['logpval'])) for i in range(len(pathway_enrichment_df['logpval']))]
    plt.figure(figsize=(8, 6))  # 设置图形大小
    plt.barh(pathway_enrichment_df.index, pathway_enrichment_df['logpval'], color=colors)  # 绘制条形图
    plt.xlabel('Name')  # 设置x轴标签
    plt.ylabel('Value')  # 设置y轴标签
    plt.title('Bar Plot')  # 设置图表标题
    plt.show()


