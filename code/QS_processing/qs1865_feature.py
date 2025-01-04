import pandas as pd
import numpy as np

# 读取两个 CSV 文件
qs1865_df = pd.read_csv('../PHI_QS/NcName_QS1865.csv', header=None, names=['Sample'])
feature_df = pd.read_csv('../PHI_QS/NcName_fature.csv')

feature_df.set_index('Sample', inplace=True)

num_features = feature_df.shape[1]
merged_features = []

for sample in qs1865_df['Sample']:
    if sample in feature_df.index:
        features = feature_df.loc[sample].values
    else:
        features = np.zeros(num_features)
    merged_features.append(features)

merged_features_df = pd.DataFrame(merged_features, columns=feature_df.columns)

merged_features_df.insert(0, 'Sample', qs1865_df['Sample'])

# 保存合并后的数据框到新的 CSV 文件
merged_features_df.to_csv('../PHI_QS/merged_QS1865_features.csv', index=False)

