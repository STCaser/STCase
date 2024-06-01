import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import scanpy as sc
import math
from scipy.spatial.distance import cdist
import pandas as pd
import seaborn as sns
import os, glob
import shutil
import igraph as ig
import random
from collections import Counter
from plotnine import *
import pickle
from scipy.sparse import csr_matrix
from tqdm import tqdm
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell
from pyscenic.binarization import binarize
from pyscenic.binarization import derive_threshold

np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告




def detect_outliers_z_score(data, 
                            threshold=3):
    """
    使用Z-score方法检测outlier
    """
    z_scores = (data - np.mean(data)) / np.std(data)
    return np.where(np.abs(z_scores) > threshold)[0]

def find_radius(adata,  
                scope=6):
    ## 计算每个点周围是scope个邻居的时候，半径是多少
    n = adata.shape[0]
    coords = adata.obsm['spatial']
    distances = cdist(coords, coords)
    nearby_dis = []
    for i in range(n):
        sorted_indices = np.argsort(distances[i])
        nearby_indices = sorted_indices[1:scope + 1]  # 第0个是它自己，不需要计算
        nearby_dis.append([distances[i][a] for a in nearby_indices])
    new_nearby_dis = np.concatenate(nearby_dis)
    # 示例
    outliers = detect_outliers_z_score(new_nearby_dis)
    new_nearby_dis = np.delete(new_nearby_dis, outliers)
    radius = np.max(new_nearby_dis)
    print('The radius is: ' + str(radius))
    adata.uns['radius'] = {'radius':radius}
    
    
def subset_anndata(adata, cell_type_list, key='cell_type'):
    return adata[adata.obs[key].isin(cell_type_list)]

def check_elements(list1, 
                   list2, 
                   number):
    if number == 'all':
        ## 检查list1是否全部存在于list2中，返回True/False
        set2 = set(list2)
        return set2.issuperset(list1)
    else:
        set1 = set(list1)
        set2 = set(list2)
        return len(set1.intersection(set2)) >= number


def grep_exist_LR(adata,
                  DB_interaction,
                  DB_complex,
                  if_hvg=True,
                  n_top_genes=None,
                  min_mean=0.0125,
                  max_mean=3,
                  min_disp=0.5,
                  max_disp=np.inf):
    ## 检查Database中的LR对对应的基因是否都存在于adata的var中，如果一个LR对中有一个基因不存在于adata的var中，则删除
    adata.uns['DB_complex'] = DB_complex
    
    DB_complex_index = DB_complex.index.tolist()
    LR_noexist = []
    if if_hvg == True:
        sc.pp.highly_variable_genes(adata, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, max_disp=max_disp, n_top_genes=n_top_genes)
        hvggene = list(adata[:, adata.var.highly_variable].var.index)
    allgene = adata.var.index.tolist()
    LR_dict = {}
    
    for i in DB_interaction.index:
        list_tmp = []
        ligand = DB_interaction.loc[i,'ligand']
        receptor = DB_interaction.loc[i,'receptor']
        ligand_annotation = DB_interaction.loc[i,'ligand_annotation']

        if ligand in DB_complex_index:
            sub = DB_complex.loc[ligand].tolist()
            ligands_sub = [i for i in sub if str(i) != 'nan']
        else:
            ligands_sub = ligand
        if receptor in DB_complex_index:
            sub = DB_complex.loc[receptor].tolist()
            receptor_sub = [i for i in sub if str(i) != 'nan']
        else:
            receptor_sub = receptor

        if type(ligands_sub) == str:
            list_tmp.append(ligands_sub)
        else:
            list_tmp = list_tmp + ligands_sub

        if type(receptor_sub) == str:
            list_tmp.append(receptor_sub)
        else:
            list_tmp = list_tmp + receptor_sub
        
        if 'pathway' in DB_interaction.columns:
            pathway_name = DB_interaction.loc[i,'pathway']
            if check_elements(list_tmp, allgene,'all'):
                if if_hvg == True:
                    if check_elements(list_tmp, hvggene, 1):
                        LR_dict[i] = {'ligand':ligand, 'receptor':receptor,'l_annotation':ligand_annotation,'pathway':pathway_name}
                else:
                    LR_dict[i] = {'ligand':ligand, 'receptor':receptor,'l_annotation':ligand_annotation,'pathway':pathway_name}
                    
        else:
            if check_elements(list_tmp, allgene,'all'):
                if if_hvg == True:
                    if check_elements(list_tmp, hvggene, 1):
                        LR_dict[i] = {'ligand':ligand, 'receptor':receptor,'l_annotation':ligand_annotation}
                else:
                    LR_dict[i] = {'ligand':ligand, 'receptor':receptor,'l_annotation':ligand_annotation}
            
    adata.uns['LR_pair_information'] = LR_dict
    
    
    geneLR = DB_interaction.loc[list(LR_dict.keys())]['ligand'].tolist() + DB_interaction.loc[list(LR_dict.keys())]['receptor'].tolist()
    #geneall = DB_geneinfo['Symbol'].tolist()
    complexset = list(set([lr for lr in geneLR if lr in list(DB_complex.index)]))
    geneset = list(set(geneLR) - set(complexset))
    DB_complex_keep = DB_complex.loc[complexset]
    complex_sub = DB_complex_keep.stack().tolist()
    complex_sub = list(set(complex_sub))
    geneLR = list(set(complex_sub + geneset))
    
    LR_gene_com_keep = {'geneset':geneset,
                       'complexset':complexset,
                       'complex_sub':complex_sub,
                       'geneall':geneLR}
    
    adata.uns['LR_gene_complex_information'] = LR_gene_com_keep
    print('The number of keep LR pair is '+str(len(LR_dict.keys())))
    
def progress_bar(progress, names):
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percentage = int(progress * 100)
    print(f'{names}: |{bar}| {percentage}%', end="\r")

def gmean(expression_levels):
    # 计算四分位数
    q1 = np.percentile(expression_levels, 25)
    q2 = np.percentile(expression_levels, 50)
    q3 = np.percentile(expression_levels, 75)
    # 计算平均值
    average = (1/2) * q2 + (1/4) * (q1 + q3)
    return average

def get_gene_expression(mat,
                        gene_index, ## index 和 gene的对应关系
                        gene,
                        dic,
                        key_name=None):
    ## 输入一个gene，获取其对应的基因表达值
    ## 如果输入的是一个list，则获取几个基因的几何平均表达
    if isinstance(gene, str):
        dic[gene] = mat[:, gene_index[gene]]
    elif isinstance(gene, list):
        gene_indices = np.array([gene_index[g] for g in gene])
        mat_sub = mat[:, gene_indices]
        dic[key_name] = np.apply_along_axis(gmean, axis=1, arr=mat_sub)

        
def get_LR_gene_exp_pool(mat, gene_index, all_complex, geneLR_exp_dict, DB_complex):
    """模拟进程池"""
    while True:
        try:
            com = all_complex.get(False)
            sub = DB_complex.loc[com].tolist()
            receptor_sub = [i for i in sub if str(i) != 'nan']
            get_gene_expression(mat,gene_index, receptor_sub, geneLR_exp_dict, key_name=com, )
            
        except Exception:
            if all_complex.empty():
                break
    
                
def get_LR_gene_exp(adata,
                    threads=40):
    ## 该函数计算了所有受体和配体的基因表达，如果受配体是复合物，则使用几何平均值计算平均表达。
    ## 返回一个dict，里面是受体配体基因或复合物的每个细胞的表达值
    
    geneset = adata.uns['LR_gene_complex_information']['geneset']
    complexset = adata.uns['LR_gene_complex_information']['complexset']
    DB_complex = adata.uns['DB_complex']
    
    try:
        mat = adata.X.toarray()
    except AttributeError:
        mat = adata.X
    gl = adata.var.index
    gene_index = {name: i for i, name in enumerate(gl)}
    geneLR_exp_dict = {}
    for gene in geneset:
        get_gene_expression(mat, gene_index, gene, geneLR_exp_dict)
    p_list = []
    # 先将多进程所要执行的任务的所有参数放入队列中
    all_task = multiprocessing.Queue()
    for com in complexset:
        all_task.put(com)
    # 结果存储
    com_dict = multiprocessing.Manager().dict()
    
    # 启动多进程

    for i in range(threads):
        p = multiprocessing.Process(target=get_LR_gene_exp_pool, args=(mat, gene_index, all_task, com_dict, DB_complex,))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    
    com_dict = dict(com_dict)
    geneLR_exp_dict.update(com_dict)
    
    adata.uns['LR_gene_complex_exp'] = geneLR_exp_dict
    
    
def get_close_gene(adata,
                   background_number=100):
    ## 找出与每个受体配体基因或复合物的平均表达值最接近的100个close基因，然后计算这些基因的表达值
    ## 返回两个dict，一个是每个受体配体基因或复合物最接近的100个close基因，第二个是这些close基因的平均表达
    geneLR_exp_dict = adata.uns['LR_gene_complex_exp']
    
    try:
        mat = adata.X.toarray()
    except AttributeError:
        mat = adata.X
    gl = adata.var.index
    gene_index = {name: i for i, name in enumerate(gl)}
    
    
    geneLR = adata.uns['LR_gene_complex_information']['geneall']
    
    all_genes = [item for item in adata.var_names.tolist()
                 if not (item.startswith("MT-") or item.startswith("MT_"))]
    geneLR = [item for item in geneLR
                 if not (item.startswith("MT-") or item.startswith("MT_"))]
    means = adata.to_df()[all_genes].mean().sort_values()

    geneLR_close_means_gene = {}
    for key in geneLR_exp_dict.keys():
        means_exp = np.mean(geneLR_exp_dict[key])
        selected_lf = (abs(means - means_exp).sort_values().drop(geneLR)[:background_number].index.tolist())
        random.shuffle(selected_lf)
        geneLR_close_means_gene[key] = selected_lf

    close_gene_list = []
    for key in geneLR_close_means_gene.keys():
        close_gene_list = close_gene_list + geneLR_close_means_gene[key]
    close_gene_list = list(set(close_gene_list))

    close_gene_exp_dict = {}
    for gene in close_gene_list:
        get_gene_expression(mat,gene_index, gene, close_gene_exp_dict)
        
    adata.uns['LR_close_gene'] = geneLR_close_means_gene
    adata.uns['LR_close_gene_exp'] = close_gene_exp_dict
    #del(adata.uns['LR_gene_complex_information'])
    
def get_true_weight_matirx(adata,
                           method, ## 计算lr分数的方法
                           scope=6, ## 非分泌性受配体扩散范围到其周围的几个点
                           min_exp=0.01):
    
    n = adata.shape[0]
    coords = adata.obsm['spatial']
    distances = cdist(coords, coords)
    adata.obsm['distances'] = distances
    radius = adata.uns['radius']['radius']
    
    LR_weight_dict = {}
    #LR_neighbors_dict = {}
    LR_dict = adata.uns['LR_pair_information']
    
    geneLR_exp_dict = adata.uns['LR_gene_complex_exp']
    #del(adata.uns['LR_gene_complex_exp'])
    
    for key in LR_dict.keys():
        LR_weight_dict[key] = np.zeros((n, n),dtype='float32')
        #LR_neighbors_dict[key] = np.zeros((n, n))

    LR_secreted_dict = {}
    LR_unsecreted_dict = {}
    for key in LR_dict.keys():
        if LR_dict[key]['l_annotation'] == 'Secreted':
            LR_secreted_dict[key] = LR_dict[key]
        else:
            LR_unsecreted_dict[key] = LR_dict[key]

    Wr = (3*radius/2)
    print('Now processing the unsecreted')
    for i in range(n):
        neighbors = np.where(distances[i] < radius)[0]
        #neighbors = np.delete(neighbors, np.where(neighbors == i))
        if len(neighbors) == 0:
            continue
        distance_factor = np.exp(-(distances[i, neighbors] / Wr)**2)  # 距离衰减因子，使用指数函数
        for key in LR_unsecreted_dict.keys():
            ligand_name = LR_unsecreted_dict[key]['ligand']
            receptor_name = LR_unsecreted_dict[key]['receptor']
            ligand = geneLR_exp_dict[ligand_name]
            receptor = geneLR_exp_dict[receptor_name]

            score_l = ligand[i] * distance_factor
            indx = np.where(score_l < min_exp)[0]
            neighbors_nonsig = neighbors[indx]
            neighbors_sig = np.delete(neighbors, indx)
            LR_weight_dict[key][i, neighbors_nonsig] = 0
            distance_factor_sig = np.delete(distance_factor, indx)
            if method == 'Hill':
                scores_tmp = ligand[i] * receptor[neighbors_sig] * distance_factor_sig
                scores = scores_tmp / (0.5+scores_tmp)
            if method == 'normal':
                scores = ligand[i] * receptor[neighbors_sig] * distance_factor_sig
            if method =='square_root':
                scores_tmp = ligand[i] * receptor[neighbors_sig] * distance_factor_sig
                scores = np.sqrt(scores_tmp)
            
            non_zero_positions = np.where(scores != 0)
            neighbors_sig_nozero = neighbors_sig[non_zero_positions]
            LR_weight_dict[key][i, neighbors_sig] = scores
            #LR_neighbors_dict[key][i, neighbors_sig_nozero] = 1
        progress_bar((i+1)/n, 'Computed weight matrix process')
    
    for key in LR_unsecreted_dict.keys():
        LR_weight_dict[key] = csr_matrix(LR_weight_dict[key])
        #LR_neighbors_dict[key] = csr_matrix(LR_neighbors_dict[key])

    print('\nNow processing the secreted')
    num = 1 
    for key in LR_secreted_dict.keys():
        ligand_name = LR_dict[key]['ligand']
        receptor_name = LR_dict[key]['receptor']
        ligand = geneLR_exp_dict[ligand_name]
        receptor = geneLR_exp_dict[receptor_name]
        distance_radius = np.where(ligand == 0, 0, np.sqrt(-np.log(min_exp/ligand)) * Wr)
        for i in range(n):
            neighbors = np.where(distances[i] < distance_radius[i])[0]
            #neighbors = np.delete(neighbors, np.where(neighbors == i))
            if len(neighbors) == 0:
                continue
            distance_factor = np.exp(-(distances[i, neighbors] / Wr)**2)  # 距离衰减因子，使用指数函数
            
            if method == 'Hill':
                scores_tmp = ligand[i] * receptor[neighbors] * distance_factor
                scores = scores_tmp / (0.5+scores_tmp)
            if method == 'normal':
                scores = ligand[i] * receptor[neighbors] * distance_factor
            if method =='square_root':
                scores_tmp = ligand[i] * receptor[neighbors] * distance_factor
                scores = np.sqrt(scores_tmp)
                
            non_zero_positions = np.where(scores != 0)
            neighbors_nozero = neighbors[non_zero_positions]
            LR_weight_dict[key][i, neighbors] = scores
            #LR_neighbors_dict[key][i, neighbors_nozero] = 1
            
        LR_weight_dict[key] = csr_matrix(LR_weight_dict[key])
        #LR_neighbors_dict[key] = csr_matrix(LR_neighbors_dict[key])
        progress_bar(num/len(LR_secreted_dict.keys()), 'Computed weight matrix process')
        num = num + 1
    
    adata.uns['LR_cell_weight'] = LR_weight_dict
    #adata.uns['Cell_neighbors'] = LR_neighbors_dict
    
    
def find_key_position(dictionary, key):
    keys = list(dictionary.keys())
    if key in keys:
        position = keys.index(key)
        return position
    else:
        return None
    
def calculate_fake_weight(adata, 
                          LR_name, 
                          result_dict,
                          method,
                          percent_list,
                          all_task_number,
                          min_exp=0.01,
                          cutoff=0.05):
    
    n = adata.shape[0]
    radius = adata.uns['radius']['radius']
    distances = adata.obsm['distances']
    Wr = (3*radius/2)
    
    geneLR_close_means_gene = adata.uns['LR_close_gene']
    close_gene_exp_dict = adata.uns['LR_close_gene_exp']
    LR_inf_dict = adata.uns['LR_pair_information']
    LR_true_weight_dict = adata.uns['LR_cell_weight']

    ligand_fake_name = geneLR_close_means_gene[LR_inf_dict[LR_name]['ligand']]
    receptor_fake_name = geneLR_close_means_gene[LR_inf_dict[LR_name]['receptor']]
    
    background_number = len(ligand_fake_name)
    LR_fake_weight_array = np.zeros((background_number, n, n),dtype='float32')
    
    if LR_inf_dict[LR_name]['l_annotation'] == 'Secreted':
        for num in range(background_number):
            ligand_name = ligand_fake_name[num]
            receptor_name = receptor_fake_name[num]
            ligand = close_gene_exp_dict[ligand_name]
            receptor = close_gene_exp_dict[receptor_name]
            distance_radius = np.where(ligand == 0, 0, np.sqrt(-np.log(min_exp/ligand)) * Wr)
            for i in range(n):
                neighbors = np.where(distances[i] < distance_radius[i])[0]
                #print(len(neighbors))
                if len(neighbors) == 0:
                    continue
                #neighbors = np.delete(neighbors, np.where(neighbors == i))
                distance_factor = np.exp(-(distances[i, neighbors] / Wr)**2)  # 距离衰减因子，使用指数函数
                
                if method == 'Hill':
                    scores_tmp = ligand[i] * receptor[neighbors] * distance_factor
                    scores = scores_tmp / (0.5+scores_tmp)
                if method == 'normal':
                    scores = ligand[i] * receptor[neighbors] * distance_factor
                if method =='square_root':
                    scores_tmp = ligand[i] * receptor[neighbors] * distance_factor
                    scores = np.sqrt(scores_tmp)
                
                LR_fake_weight_array[num][i, neighbors] = scores
        
        weight_matrix_pval = 1 - (np.sum(LR_fake_weight_array < LR_true_weight_dict[LR_name].toarray().reshape(1, n, n), axis=0) / background_number)
        
    else:
        for i in range(n):
            neighbors = np.where(distances[i] < radius)[0]
            #neighbors = np.delete(neighbors, np.where(neighbors == i))
            distance_factor = np.exp(-(distances[i, neighbors] / Wr)**2)  # 距离衰减因子，使用指数函数
            for num in range(100):
                ligand_name = ligand_fake_name[num]
                receptor_name = receptor_fake_name[num]
                ligand = close_gene_exp_dict[ligand_name]
                receptor = close_gene_exp_dict[receptor_name]
                score_l = ligand[i] * distance_factor
                indx = np.where(score_l < min_exp)[0]
                neighbors_nonsig = neighbors[indx]
                neighbors_sig = np.delete(neighbors, indx)
                LR_fake_weight_array[num][i, neighbors_nonsig] = 0
                distance_factor_sig = np.delete(distance_factor, indx)
                
                if method == 'Hill':
                    scores_tmp = ligand[i] * receptor[neighbors_sig] * distance_factor_sig
                    scores = scores_tmp / (0.5+scores_tmp)
                if method == 'normal':
                    scores = ligand[i] * receptor[neighbors_sig] * distance_factor_sig
                if method =='square_root':
                    scores_tmp = ligand[i] * receptor[neighbors_sig] * distance_factor_sig
                    scores = np.sqrt(scores_tmp)
                
                LR_fake_weight_array[num][i, neighbors_sig] = scores

        weight_matrix_pval = 1 - (np.sum(LR_fake_weight_array < LR_true_weight_dict[LR_name].toarray().reshape(1, n, n), axis=0) / background_number)
    
    #adata.uns['LR_cell_weight'][LR_name][weight_matrix_pval > cutoff] = 0
    result_dict[LR_name] = weight_matrix_pval
    pos = find_key_position(result_dict, LR_name)
    length = pos + 1
    percent = length/all_task_number
    all_task_number = all_task_number
    percent_list.append(percent)
    for cf in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        if (percent >= cf) & all(num < cf for num in percent_list[:length-1]):
            progress_bar(percent, 'Permutation test process')
            break
def permutation_test_pool(adata, 
                          all_LR_name, 
                          result_dict,
                          method,
                          min_exp,
                          cutoff,
                          percent_list,
                          all_task_number):
    """模拟进程池"""
    while True:
        try:
            LR_name = all_LR_name.get(False)
            calculate_fake_weight(adata, LR_name, result_dict, method, percent_list, all_task_number, min_exp, cutoff)
        except Exception:
            if all_LR_name.empty():
                break

                
def permutation_test(adata, 
                     method, 
                     threads=40,
                     min_exp=0.01,
                     cutoff=0.05):
    p_list = []
    # 先将多进程所要执行的任务的所有参数放入队列中
    all_task = multiprocessing.Queue()
    LR_dict = adata.uns['LR_pair_information']
    all_task_number = len(LR_dict.keys())
    for LR_name in LR_dict.keys():
        all_task.put(LR_name)
    # 结果存储
    result_dict = multiprocessing.Manager().dict()
    percent_list = multiprocessing.Manager().list()
    # 启动多进程
    for i in range(threads):
        p = multiprocessing.Process(target=permutation_test_pool, args=(adata, all_task, result_dict, method, min_exp, cutoff, percent_list, all_task_number,))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    print('\nNow filter low confidence value')
    fake_LR_pval = dict(result_dict)
    num = 1
    LR_neighbors_dict = {}
    for LR_name in adata.uns['LR_cell_weight'].keys():
        tmp_weight = adata.uns['LR_cell_weight'][LR_name].toarray()
        tmp_weight[fake_LR_pval[LR_name] >= 0.05] = 0
        adata.uns['LR_cell_weight'][LR_name] = csr_matrix(tmp_weight)
        for i in range(len(tmp_weight)):
            for j in range(len(tmp_weight[i])):
                if tmp_weight[i][j] != 0:
                    tmp_weight[i][j] = 1
        LR_neighbors_dict[LR_name] = csr_matrix(tmp_weight)
        percent = num / len(adata.uns['LR_cell_weight'].keys())
        progress_bar(percent, 'Filter low confidence value process')
        num = num + 1
    #adata.uns['LR_cell_pval'] = fake_LR_pval
    adata.uns['Cell_neighbors'] = LR_neighbors_dict
        
        
def run_scenic(adata, DB_interaction, DATABASES_GLOB, MOTIF_ANNOTATIONS_FNAME):
    exp_matrix = pd.DataFrame(adata.X.toarray(), columns=adata.var.index, index=adata.obs.index)
    tf_names = []
    for i in range(len(list(DB_interaction['TF']))):
        if type(list(DB_interaction['TF'])[i]) is not float:
            tf_names = tf_names + list(DB_interaction['TF'])[i].split(', ') 
    tf_names = list(set(tf_names))
    db_fnames = glob.glob(DATABASES_GLOB)
    def name(fname):
        return os.path.splitext(os.path.basename(fname))[0]
    dbs = [RankingDatabase(fname=fname, name=name(fname)) for fname in db_fnames]
    # Run grnboost2
    print('Phase I: Inference of co-expression modules')
    adjacencies = grnboost2(exp_matrix, tf_names=tf_names, verbose=True)
    # Create modules from a dataframe containing weighted adjacencies between a TF and its target genes.
    modules = list(modules_from_adjacencies(adjacencies, exp_matrix, min_genes=10))
    print('Phase II: Prune modules for targets with cis regulatory footprints')
    # Calculate a list of enriched motifs and the corresponding target genes for all modules.
    df = prune2df(dbs, modules, MOTIF_ANNOTATIONS_FNAME,num_workers=20)
    regulons = df2regulons(df)
    print('Phase III: Cellular regulon enrichment matrix')
    auc_mtx = aucell(exp_matrix, regulons, num_workers=40)
    ## binarize
    thrs = []
    num = 1
    for regulon in auc_mtx.columns:
        thrs.append(derive_threshold(auc_mtx, regulon))
        progress_bar(num/len(auc_mtx.columns), 'Binarize')
        num = num + 1
    thresholds = pd.Series(index=auc_mtx.columns, data=thrs)
    auc_bin_mtx = (auc_mtx > thresholds).astype(int)
    # save result
    scenic_res_dict = {}
    scenic_res_dict['auc_mtx'] = auc_mtx
    scenic_res_dict['auc_bin_mtx'] = auc_bin_mtx
    scenic_res_dict['auc_thresholds'] = thresholds
    adata.uns['scenic_res'] = scenic_res_dict
    adata.uns['scenic_res']['auc_thresholds'] = dict(adata.uns['scenic_res']['auc_thresholds'])


def add_intracellular_signals(adata, DB_interaction, DATABASES_GLOB, MOTIF_ANNOTATIONS_FNAME, if_stringent):
	print('Run SCENIC')
	run_scenic(adata, DB_interaction, DATABASES_GLOB, MOTIF_ANNOTATIONS_FNAME)
	auc_mtx = adata.uns['scenic_res']['auc_mtx']
	auc_mtx = auc_mtx.loc[adata.obs.index]
	auc_bin_mtx = adata.uns['scenic_res']['auc_bin_mtx']
	thresholds = adata.uns['scenic_res']['auc_thresholds']
	auc_mtx[auc_mtx < thresholds] = 0
	regulons_list = [r.split('(')[0] for r in auc_mtx.columns]
	regulons_dict = {}
	for lr in DB_interaction.index:
		if type(DB_interaction.loc[lr,'TF']) is not float:
			if len(list(set(DB_interaction.loc[lr,'TF'].split(', ')) & set(regulons_list))) > 0:
				regulons_dict[lr] = list(set(DB_interaction.loc[lr,'TF'].split(', ')) & set(regulons_list))
	if if_stringent:
		num = 1
		for lr in adata.uns['LR_cell_weight'].keys():
			if lr in regulons_dict.keys():
				lr_weight = adata.uns['LR_cell_weight'][lr].toarray()
				auc_mtx_lr = auc_mtx[[r+'(+)' for r in regulons_dict[lr]]]
				row_sums = auc_mtx_lr.sum(axis=1)
				transformed_sums = row_sums.apply(lambda x: x / (x + 0.5))
				transformed_sums = np.expand_dims(transformed_sums, axis=1)
				multiplier_str = np.where(transformed_sums>0, transformed_sums+1, transformed_sums)
				lr_weight_str = lr_weight * (multiplier_str.T)
				adata.uns['LR_cell_weight'][lr] = csr_matrix(lr_weight_str)
			else:
				lr_weight = adata.uns['LR_cell_weight'][lr].toarray()
				adata.uns['LR_cell_weight'][lr] = csr_matrix(np.zeros(lr_weight.shape))
			progress_bar(num/len(adata.uns['LR_cell_weight'].keys()), 'Add intracellular signals')
			num = num + 1
	else:
		num = 1
		for lr in adata.uns['LR_cell_weight'].keys():
			if lr in regulons_dict.keys():
				lr_weight = adata.uns['LR_cell_weight'][lr].toarray()
				auc_mtx_lr = auc_mtx[[r+'(+)' for r in regulons_dict[lr]]]
				row_sums = auc_mtx_lr.sum(axis=1)
				transformed_sums = row_sums.apply(lambda x: x / (x + 0.5))
				transformed_sums = np.expand_dims(transformed_sums, axis=1)
				multiplier_nonstr = transformed_sums + 1
				lr_weight_nonstr = lr_weight * (multiplier_nonstr.T)
				adata.uns['LR_cell_weight'][lr] = csr_matrix(lr_weight_nonstr)
			progress_bar(num/len(adata.uns['LR_cell_weight'].keys()), 'Add intracellular signals')
			num = num + 1
	adata.uns['scenic_res']['regulons_dict'] = regulons_dict
    
    
    
def filter_cell(adata, 
                key,
                if_self=True):
    ## 过滤掉没有与其他细胞有相互作用的细胞
    ## 同时也可以过滤自分泌的细胞，如果只关注细胞间相互作用的话
    n = adata.shape[0]
    LR_cell_weight = adata.uns['LR_cell_weight']
    cw = np.zeros((n, n),dtype='float32')
    for key in LR_cell_weight.keys():
        cw = cw + LR_cell_weight[key].toarray()
    
    if if_self == False:
        obs_meta = adata.obs.copy()
        obs_meta.index = range(n)
        unique_values = obs_meta[key].unique()
        for ct in unique_values:
            index_ct = list(obs_meta.index[obs_meta[key]==ct])
            mask = np.zeros_like(cw, dtype=bool)
            # 将指定行和列的布尔值设置为True
            for idx in index_ct:
                mask[index_ct,idx] = True
            # 将布尔数组中对应位置为True的元素设置为0
            cw[mask] = 0
    
    index_keep = []
    for i in range(n):
        score_cell = np.sum(cw[:, i]) + np.sum(cw[i, :])
        if score_cell != 0:
            index_keep.append(i)

    for key in adata.uns['LR_cell_weight']:
        cw = adata.uns['LR_cell_weight'][key].toarray()
        cn = adata.uns['Cell_neighbors'][key].toarray()
        cw_sel = cw[index_keep, :]
        cw_sel = cw_sel[:, index_keep]
        cn_sel = cn[index_keep, :]
        cn_sel = cn_sel[:, index_keep]
        adata.uns['LR_cell_weight'][key] = csr_matrix(cw_sel) 
        adata.uns['Cell_neighbors'][key] = csr_matrix(cn_sel) 

    return adata[index_keep,]

def aggregate_matrix(adata,
                     key):
    LR_dict = adata.uns['LR_pair_information']
    LR_weight_dict = adata.uns['LR_cell_weight']
    LR_neighbors_dict = adata.uns['Cell_neighbors']
    
    ct_interaction_dict = {}
    ct_interaction_per_dict = {}
    ct_interaction_edgenum_dict = {}
    cell_type_list = adata.obs[key].tolist()
    cell_type = list(set(cell_type_list))
    adata.uns[key+'_list'] = cell_type
    ct_n = len(cell_type)
    for key in LR_weight_dict.keys():
        weight_direct_matrix =  LR_weight_dict[key].toarray()
        neighbors_direct_matrix =  LR_neighbors_dict[key].toarray()
        interaction = np.zeros((ct_n, ct_n))
        interaction_per = np.zeros((ct_n, ct_n))
        interaction_edgenum = np.zeros((ct_n, ct_n))
        for l in range(ct_n):
            l_ct = cell_type[l]
            index_l = np.where(np.array(cell_type_list)==l_ct)
            weight_direct_matrix_tmp = weight_direct_matrix[index_l]
            neighbors_direct_matrix_tmp = neighbors_direct_matrix[index_l]
            merged_row = np.sum(weight_direct_matrix_tmp, axis=0)
            neighbors_merged_row = np.sum(neighbors_direct_matrix_tmp, axis=0)
            for r in range(ct_n):
                r_ct = cell_type[r]
                interaction[l,r] = np.sum(merged_row[np.where(np.array(cell_type_list)==r_ct)])
                edge_num = np.sum(neighbors_merged_row[np.where(np.array(cell_type_list)==r_ct)])
                if edge_num == 0:
                    interaction_per[l,r] = interaction[l,r]
                else:
                    interaction_per[l,r] = interaction[l,r]/edge_num
                interaction_edgenum[l,r] = edge_num
        ct_interaction_dict[key] = interaction
        ct_interaction_per_dict[key] = interaction_per
        ct_interaction_edgenum_dict[key] = interaction_edgenum
    
    ## 记录针对每个LR对，不同细胞类型之间的相互作用程度
    adata.uns['LR_celltype_weight'] = ct_interaction_dict
    adata.uns['LR_celltype_mean_weight'] = ct_interaction_per_dict
    adata.uns['LR_celltype_edge_num'] = ct_interaction_edgenum_dict
    
    result_weight = np.zeros_like(next(iter(ct_interaction_dict.values())))
    result_weight_per = np.zeros_like(next(iter(ct_interaction_per_dict.values())))
    result_count = np.zeros_like(next(iter(ct_interaction_dict.values())))
    result_edge = np.zeros_like(next(iter(ct_interaction_edgenum_dict.values())))
    for array in ct_interaction_dict.values():
        result_weight += array
    for array in ct_interaction_dict.values():
        result_count += (array != 0)
    for array in ct_interaction_per_dict.values():
        result_weight_per += array

    # 提取ct_interaction_edgenum_dict字典中每个array对应位置最大的值，组成新的数组，目的是看每个细胞类型之间有多少相互作用的细胞对
    edge_arrays = list(ct_interaction_edgenum_dict.values())
    result_edge = np.mean(edge_arrays,axis=0)
    result_edge = np.array(result_edge)
    
    adata.uns['LR_celltype_aggregate_weight'] = {'weight':result_weight,
                                                 'weight_per':result_weight_per,
                                                 'count':result_count,
                                                 'edge_num':result_edge}
    
    #celltype_pathway_level_cal
    if 'pathway' in list(LR_dict[list(LR_dict.keys())[0]].keys()):
        pathway_list = list(set([LR_dict[lr]['pathway'] for lr in LR_dict.keys()]))
        pathway_list = [p for p in pathway_list if str(p) != 'nan']
        pathway_interaction_mean_weight_dict = {}
        pathway_interaction_weight_dict = {}
        pathway_interaction_count_dict = {}
        pathway_interaction_edge_dict = {}
        for pathway in pathway_list:
            pathway_lr = [lr for lr in LR_dict.keys() if LR_dict[lr]['pathway'] == pathway]

            LR_pathway_mean_dict = {lr:adata.uns['LR_celltype_mean_weight'][lr] for lr in pathway_lr}
            LR_pathway_dict = {lr:adata.uns['LR_celltype_weight'][lr] for lr in pathway_lr}
            LR_pathway_edge_dict = {lr:adata.uns['LR_celltype_edge_num'][lr] for lr in pathway_lr}
            presult_weight_mean = np.zeros_like(next(iter(LR_pathway_mean_dict.values())))
            presult_weight = np.zeros_like(next(iter(LR_pathway_dict.values())))
            presult_edge = np.zeros_like(next(iter(LR_pathway_edge_dict.values())))
            presult_count = np.zeros_like(next(iter(LR_pathway_dict.values())))

            for array in LR_pathway_mean_dict.values():
                presult_weight_mean += array
            pathway_interaction_mean_weight_dict[pathway] = presult_weight_mean

            for array in LR_pathway_dict.values():
                presult_weight += array
            pathway_interaction_weight_dict[pathway] = presult_weight
            
            for array in LR_pathway_dict.values():
                presult_count += (array != 0)
            pathway_interaction_count_dict[pathway] = presult_count
            
            # 提取ct_interaction_edgenum_dict字典中每个array对应位置最大的值，组成新的数组，目的是看每个细胞类型之间有多少相互作用的细胞对
            edge_arrays = list(LR_pathway_edge_dict.values())
            presult_edge = np.mean(edge_arrays,axis=0)
            presult_edge = np.array(presult_edge)
            pathway_interaction_edge_dict[pathway] = presult_edge

        adata.uns['LR_pathway_celltype_weight'] = pathway_interaction_weight_dict
        adata.uns['LR_pathway_celltype_mean_weight'] = pathway_interaction_mean_weight_dict
        adata.uns['LR_pathway_celltype_edge_num'] = pathway_interaction_edge_dict
        adata.uns['LR_pathway_celltype_count'] = pathway_interaction_count_dict
    del(adata.uns['DB_complex'])
    
    ##singlecell_pathway_level_cal
    def dict_array_add(dic, key=None):
        if key==None:
            key = list(dic.keys())
        # 初始化结果数组，以第一个要相加的数组为基准
        result_array = dic[key[0]].copy()
        # 遍历其他要相加的数组，并相应位置相加
        for k in key[1:]:
            result_array += dic[k]
    
        return csr_matrix(result_array)

    if 'pathway' in list(LR_dict[list(LR_dict.keys())[0]].keys()):
        pathway_list = list(set([LR_dict[lr]['pathway'] for lr in LR_dict.keys()]))
        pathway_list = [p for p in pathway_list if str(p) != 'nan']
        LR_pathway_cell_weight = {}
        
        for pw in pathway_list:
            pw_lr = [lr for lr in LR_dict.keys() if LR_dict[lr]['pathway'] == pw]
            pw_dict = {}
            for pl in pw_lr:
                pw_dict[pl] = adata.uns['LR_cell_weight'][pl].toarray() 
            if len(pw_dict) > 0:
                LR_pathway_cell_weight[pw] = dict_array_add(pw_dict)

        adata.uns['LR_pathway_cell_weight'] = LR_pathway_cell_weight



def spatial_cell_communication_run(adata, 
                                   DB_interaction,
                                   DB_complex, 
                                   method, 
                                   ct_key='cell_type',
                                   cell_type=None,
                                   if_hvg=True,   
                                   if_filter=False,
                                   if_self=True,
                                   if_intra=True,
                                   if_stringent=False,
                                   DATABASES_GLOB=None,
                                   MOTIF_ANNOTATIONS_FNAME=None,
                                   hvg_n_top_genes=None,      
                                   hvg_min_mean=0.0125, 
                                   hvg_max_mean=3,    
                                   hvg_min_disp=0.5,     
                                   hvg_max_disp=np.inf,
                                   background_number=100,
                                   threads=50, 
                                   scope=6,
                                   min_exp=0.1,
                                   cutoff=0.05):
    if if_intra:
        if DATABASES_GLOB is None or MOTIF_ANNOTATIONS_FNAME is None:
            raise ValueError("'DATABASES_GLOB' and 'MOTIF_ANNOTATIONS_FNAME' must be specified when 'if_intra' is True")
    
    print('##################################################################')
    print('Now start to calulate radius')
    find_radius(adata,scope=scope)
    
    if cell_type != None:
        print('##################################################################')
        print('Now start to subset anndata')
        adata = subset_anndata(adata, cell_type, key=ct_key)
    

    print('##################################################################')
    print('Now start to get LR pair')
    grep_exist_LR(adata,
                  DB_interaction,
                  DB_complex,
                  if_hvg=if_hvg,
                  n_top_genes=hvg_n_top_genes,
                  min_mean=hvg_min_mean,
                  max_mean=hvg_max_mean,
                  min_disp=hvg_min_disp,
                  max_disp=hvg_max_disp)

    print('##################################################################')
    print('get_LR_gene_exp')
    get_LR_gene_exp(adata,
                    threads=threads)

    print('##################################################################')
    print('get_close_gene')
    get_close_gene(adata,
                   background_number=background_number)

    print('##################################################################')
    print('Now start to get true weight matirx')
    get_true_weight_matirx(adata,
                           method, ## 计算lr分数的方法
                           scope=scope, ## 非分泌性受配体扩散范围到其周围的几个点
                           min_exp=min_exp)

    print('\n##################################################################')
    print('Now start to permutation test')
    permutation_test(adata, 
                     method, 
                     threads=threads,
                     min_exp=min_exp,
                     cutoff=cutoff)
    if if_intra:
        print('\n##################################################################')
        print('Now  start to add intracellular signals')
        add_intracellular_signals(adata, DB_interaction, DATABASES_GLOB, MOTIF_ANNOTATIONS_FNAME, if_stringent)
    if if_filter:
        print('\n##################################################################')
        print('Now  start to filter cell')
        adata = filter_cell(adata, key=ct_key, if_self=if_self)
        print('##################################################################')
    else:
        print('\n##################################################################')
    print('Now  start to aggregate')
    aggregate_matrix(adata, ct_key)
    print('Spatial cell communication finished!')
    return adata
