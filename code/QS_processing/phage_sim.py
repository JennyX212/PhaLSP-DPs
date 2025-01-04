# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
#
#
# df_162 = pd.read_csv('../data/QS_data/QS_sim_data/phage162_feature.csv')
# df_112 = pd.read_csv('../data/QS_data/QS_sim_data/phage112_feature.csv')
#
# # 2. 提取样本名称和特征向量
# # 假设第一列是样本名称，后面是256维特征
# samples_162 = df_162.iloc[:, 0].tolist()
# features_162 = df_162.iloc[:, 1:].values
#
# samples_112 = df_112.iloc[:, 0].tolist()
#
# similarity_matrix = cosine_similarity(features_162, features_112)
#
# similarity_df = pd.DataFrame(similarity_matrix, index=samples_162, columns=samples_112)
#
# similarity_df.reset_index(inplace=True)
# similarity_df.rename(columns={'index': 'Sample'}, inplace=True)
#
# output_csv_path = '../data/QS_data/QS_sim_data/cosine_similarity_162_112.csv'
# similarity_df.to_csv(output_csv_path, index=False)
#
# print(f"相似性矩阵已成功保存到 {output_csv_path}")


import pandas as pd
def add_max_similarity_column(input_csv_path, output_csv_path):

    try:
        df = pd.read_csv(input_csv_path, index_col=0)
        print(f"成功读取CSV文件: {input_csv_path}")
    except FileNotFoundError:
        print(f"文件未找到，请检查路径是否正确：{input_csv_path}")
        return

    num_rows, num_cols = df.shape
    print(f"数据形状: {num_rows} 行, {num_cols} 列")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        print("警告: 某些列包含非数值型数据，可能影响相似性计算。")

    df['Max_Similarity_Sample'] = df.idxmax(axis=1)


    df_reset = df.reset_index()
    df_reset.rename(columns={'index': 'Sample'}, inplace=True)

    print(f"新列 'Max_Similarity_Sample' 已添加。")

    # 保存到新的CSV文件
    try:
        df_reset.to_csv(output_csv_path, index=False)
        print(f"新CSV文件已保存到: {output_csv_path}")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")

input_csv = '../data/QS_data/QS_sim_data/cosine_similarity_162_112.csv'
output_csv = '../data/QS_data/QS_sim_data/cosine_similarity_162_112_with_max.csv'
add_max_similarity_column(input_csv, output_csv)
