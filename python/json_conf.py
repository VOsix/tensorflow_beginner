# -*- coding: utf-8 -*-
# Author: lzjiang
import json

feature_map_file = '../tensorflow/basic_model/feature_map.json'
with open(feature_map_file, 'r') as f:
    feature_info = json.load(f)

column_list = feature_info["columns"]
column_dict = {field: index for index, field in enumerate(column_list)}

column_defaults = [['']] * len(column_dict)
for col in column_list:
    column_defaults[column_dict[col]] = [0.0]

print(column_list, "\n", column_defaults, "\n", column_dict.items())
feature_dict = {}
for key, value in column_dict.items():
    print(key, value)
