# -*- coding: utf-8 -*-


# import os

# path = './temp/json'
# json_file = os.listdir(path)
# os.system("activate labelme")
# for file in json_file:
#     os.system("labelme_json_to_dataset.exe %s"%(path + '/' + file))

import os
path = 'temp/json'  # path为json文件存放的路径
json_file = os.listdir(path)
for file in json_file:
    os.system("python C:/Users/cl\Anaconda3/envs/pytorch_gpu/Scripts/labelme_json_to_dataset.exe %s"%(path + '/' + file))