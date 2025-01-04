import pandas as pd
import numpy as np

s_list = []
s = ''
k_mer = 3
# 间隔几个字符
d = 3

def load_fa(path):
    """a function to read fasta file from the path and store in a dict"""
    genes_seq = {}
    with open(path,"r") as sequences:
        lines = sequences.readlines()

    for line in lines:
        if line.startswith(">"):
            genename = line.split()[0]
            genes_seq[genename] = ''
        else:
            genes_seq[genename] += line.strip()
    return genes_seq

def backtrack(str, index):
    global s
    if index == k_mer:
        s_list.append(s)
        return
    for i in range(0, len(str)):
        s = s + str[i]
        backtrack(str, index + 1)
        s = s[:-1]

def cheak(str):
    for c in str:
        if c not in ('A', 'T', 'C', 'G'):
            return False
    return True

backtrack('ATCG', 0)
genes_seq = load_fa(path="../data/case/crassphage.fasta")

rows = []

for gene in genes_seq.keys():
    df = pd.DataFrame(np.zeros((4 ** k_mer, 4 ** k_mer), dtype=np.int), columns=s_list, index=s_list)
    sequence = genes_seq[gene]
    for i in range(len(sequence) - k_mer * 2 - d):
        x = sequence[i:i+k_mer]
        y = sequence[i+k_mer+d:i+k_mer*2+d]
        if cheak(x) and cheak(y):
            df.loc[x, y] += 1

    df_normalized = df.div(df.sum(axis=0), axis=1)
    # 将矩阵展平为一维数组
    flattened_array = df_normalized.to_numpy().flatten()
    # 将行名（噬菌体编号）和展平后的数组合并为一个列表
    row = [gene[1:]] + flattened_array.tolist()
    rows.append(row)

# DataFrame
final_df = pd.DataFrame(rows)

final_df.to_csv("../data/case/ccrassphage_d3.csv", index=False, header=False)
