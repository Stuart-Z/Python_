# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 10:08:07 2018

@author: zhaoyang
"""
import numpy as np
import tensorflow as tf


# 输出数据
reader = tf.train.NewCheckpointReader('./save/model_1.ckpt')  
all_variables = reader.get_variable_to_shape_map()  

# 将模型中的数据转移到矩阵里面，按照变量名命名
layer1_conv1_weight = reader.get_tensor("layer1-conv1/weight")  
layer1_conv1_bias = reader.get_tensor("layer1-conv1/bias") 
layer3_conv2_weight = reader.get_tensor("layer3-conv2/weight")  
layer3_conv2_bias = reader.get_tensor("layer3-conv2/bias")  
layer5_fc1_weight = reader.get_tensor("layer5-fc1/weight")  
layer5_fc1_bias = reader.get_tensor("layer5-fc1/bias")  
layer6_fc2_weight = reader.get_tensor("layer6-fc2/weight")  
layer6_fc2_bias = reader.get_tensor("layer6-fc2/bias")  
layer7_fc3_weight = reader.get_tensor("layer7-fc3/weight")  
layer7_fc3_bias = reader.get_tensor("layer7-fc3/bias")  

