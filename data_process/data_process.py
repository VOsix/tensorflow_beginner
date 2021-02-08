# -*- coding: utf-8 -*-

# Author: lzjiang


import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

warnings.simplefilter('ignore')
pd.set_option('display.max_columns', None)

nRowsRead = None  # specify 'None' if want to read whole file
raw_sample_df = pd.read_csv('../data_taobao/raw_sample.csv', encoding='UTF-8', delimiter=',', nrows=nRowsRead)
ad_feature_df = pd.read_csv('../data_taobao/ad_feature.csv', encoding='UTF-8', delimiter=',')
user_profile_df = pd.read_csv('../data_taobao/user_profile.csv', encoding='UTF-8', delimiter=',')
raw_sample_df_new = raw_sample_df.rename(columns={"user": "userid"})
raw_sample_df.memory_usage()

df1 = raw_sample_df_new.merge(user_profile_df, on="userid")
final_df = df1.merge(ad_feature_df, on="adgroup_id")

# data_df = final_df[[
#     'pid', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation',
#     'adgroup_id', 'cate_id', 'campaign_id', 'brand', 'price',
#     'clk']]
data_df = final_df[[
    'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation',
    'clk']]
data_df.dropna()
data_df = pd.get_dummies(data_df,
                         columns=['final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation'])
print(data_df.iloc[0, :])
# data_df.to_csv('../data_taobao/taobao_train.csv', index=0)  # index为0则不保留行索引
train, test = train_test_split(data_df, test_size=0.25, random_state=1)
train.to_csv('../data_taobao/taobao_train.csv', index=0)  # index为0则不保留行索引
test.to_csv('../data_taobao/taobao_test.csv', index=0)  # index为0则不保留行索引
