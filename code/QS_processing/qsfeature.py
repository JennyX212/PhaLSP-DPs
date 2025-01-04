import pandas as pd

phqs_label = pd.read_csv('../PHI_QS/PHqs_label.csv')
hostqs_feature = pd.read_csv('../PHI_QS/hostqs_feature.csv')

host_species_index = hostqs_feature.columns.get_loc('Host species')

host_species_and_features = hostqs_feature.iloc[:, host_species_index:].copy()

host_species_and_features.set_index('Host species', inplace=True)

# 合并两个数据集，并将 hostqs_feature.csv 的特征列加到 PHqs_label.csv 中
phqs_label = phqs_label.merge(host_species_and_features, left_on='Host species', right_index=True)

# 保存合并后的结果到新的 CSV 文件
phqs_label.to_csv('../PHI_QS/merged_PHqs_label.csv', index=False)
