import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
import seaborn as sns
from scipy.stats import hypergeom

## A. inflection point finding
def Inflection_point_finding(adata,  col='weight', vis=False):
    df = adata.uns['LRP_weight']
    data = df[col].values
    data_idx = np.arange(len(data))
    kneedle = KneeLocator(data_idx,
                          data,
                          curve='convex',
                          direction='decreasing',
                          online=True)
    cutoff = max({i for i in kneedle.all_elbows if i < (data_idx[0] + data_idx[-1]) / 2})
    if vis:
        # 你的列表数据（假设是x和y坐标）
        x_values = list(range(df.shape[0]))  # 这里是x坐标的值
        y_values = list(df[col])  # 这里是y坐标的值

        # 绘制散点图
        plt.scatter(x_values, y_values,c=x_values[::-1], cmap='OrRd', s=20, linewidths=1)
        plt.axvline(x=cutoff, color='black', linestyle='--')

        # 添加标题和坐标轴标签
        plt.title('Scatter Plot')
        plt.xlabel('LR weight order')
        plt.ylabel('LR weight')

        plt.show()
    return cutoff


## B.Calculation of Specificity for CCC

def split_array_randomly(array, sizes):
    if sum(sizes) != len(array):
        raise ValueError("总数与数组长度不匹配")

    # 对数组进行随机排序
    np.random.shuffle(array)

    result = []
    index = 0
    for size in sizes:
        result.append(array[index:index + size].sum())
        index += size

    return result

def Cal_specific_ccc_single(adata, lr, subtype_list,subtype_col='subtype_label',shuffle_num=1000):
    lr_weight = adata.uns['LR_cell_weight'][lr].toarray()
    receptor_weight = lr_weight.sum(axis=0)
    cell_info = adata.obs.copy()
    cell_info.index = range(0,cell_info.shape[0])
    tumor_region_index_dict = {}
    tumor_region_rw_dict = {}
    tumor_region_len = []
    true_weight = []
    for subty in subtype_list:
        tumor_region_index_dict[subty] = cell_info[(cell_info[subtype_col] == subty)].index
        tumor_region_rw_dict[subty] = receptor_weight[tumor_region_index_dict[subty]]
        tumor_region_len.append(len(tumor_region_rw_dict[subty]))
        true_weight.append(tumor_region_rw_dict[subty].sum()/len(tumor_region_index_dict[subty]))
    
    tumor_all_rw = np.concatenate([tumor_region_rw_dict[subty] for subty in subtype_list])
    
    p_list = [0 for i in range(len(true_weight))]
    for i in range(shuffle_num):
        false_weight = split_array_randomly(tumor_all_rw, tumor_region_len)
        for idx in range(len(true_weight)):
            if true_weight[idx] > false_weight[idx]:
                p_list[idx] += 1
    p_list = [(shuffle_num - p)/shuffle_num for p in p_list]
    
    return true_weight, p_list

# 自定义归一化函数
def normalize_row(row):
    if row.max() == 0:
        return row
    else:
        return (row - row.min()) / (row.max() - row.min())
    
def Cal_specific_ccc(adata, lr_list, subtype_col='subtype_label', vmax=None, vmin=None, draw_polt=False, shuffle_num=1000, linewidths=None, linecolor=None, cutoff=0.01):
    cell_info = adata.obs.copy()
    cell_info.index = range(0,cell_info.shape[0])
    subtype_list = list(cell_info['subtype_label'].value_counts().index)
    subtype_list.remove('Others')
    
    true_weight = []
    p_list = []
    for lr in lr_list:
        true_weight_1, p_list_1 = Cal_specific_ccc_single(adata, lr, subtype_list, shuffle_num=shuffle_num)
        true_weight.append(true_weight_1)
        p_list.append(p_list_1)
    if draw_polt:
        res_weight_df = pd.DataFrame(true_weight, index=lr_list, columns=subtype_list)
        res_weight_df = res_weight_df.apply(normalize_row, axis=1)
        annot = [['' if value >= cutoff else value for value in pl] for pl in p_list]
        sns.clustermap(res_weight_df, square=True, linewidths=linewidths, linecolor=linecolor, cmap='coolwarm', annot=annot, fmt='',vmax=vmax,vmin=vmin,row_cluster=True,col_cluster=False)
        #plt.show()
    p_value_df = pd.DataFrame(p_list, index=lr_list, columns=subtype_list)
    true_weight_df = pd.DataFrame(true_weight, index=lr_list, columns=subtype_list)
    return true_weight_df, p_value_df



## C. pathway enrichment

def pathway_enrichment_for_CCC(LRP_list,DB_interaction):
    DB_interaction_subset = DB_interaction.loc[LRP_list]
    pathway_list_subset = DB_interaction_subset['pathway'].value_counts().index
    pathway_pval = pd.DataFrame(index=pathway_list_subset,columns=['pval','logpval','number_sub','number_all'])
    for pw in pathway_list_subset:
        pathway_pval.loc[pw,'pval'] = hypergeom.sf(DB_interaction_subset['pathway'].value_counts()[pw]-1,DB_interaction.shape[0],DB_interaction['pathway'].value_counts()[pw],len(LRP_list))
        pathway_pval.loc[pw,'logpval'] = -np.log10(pathway_pval.loc[pw,'pval'])
        pathway_pval.loc[pw,'number_all'] = DB_interaction['pathway'].value_counts()[pw]
        pathway_pval.loc[pw,'number_sub'] = DB_interaction_subset['pathway'].value_counts()[pw]
    pathway_pval = pathway_pval.sort_values(by='logpval', ascending=False)
    return pathway_pval
