# coding:utf-8
# training on voc

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import tensorflow as tf
from src.Data_voc_testing import Data

import numpy as np
import config

width = config.width
height = config.height
size = config.size
batch_size = config.batch_size
class_num = config.voc_class_num
if config.voc_anchors:
    anchors =  np.asarray(config.voc_anchors).astype(np.float32).reshape([-1, 3, 2])
else:
    anchors = None
iou_thresh = config.iou_thresh
prob_thresh = config.prob_thresh
score_thresh = config.score_thresh

weight_decay = config.weight_decay
cls_normalizer = config.cls_normalizer
iou_normalizer = config.iou_normalizer

lr_type = config.lr_type
lr_init = config.lr_init
lr_lower = config.lr_lower
piecewise_boundaries = config.piecewise_boundaries
piecewise_values = config.piecewise_values
optimizer_type = config.optimizer_type
momentum = config.momentum

names_file = config.voc_names
data_debug = config.data_debug
model_name = config.voc_model_name
model_path = config.voc_model_path
total_epoch = config.total_epoch
save_per_epoch = config.save_per_epoch
voc_root_dir = config.voc_root_dir


def test_val_data_draw():

    return 0

def test_data_draw():
    f_dir = '../gt_analyze/'
    
    
#     data = Data(voc_root_dir, names_file, class_num, 
#                             batch_size, anchors, is_tiny=False, size=size)

#     num=0
# #     areas = []
# #     selected_anchors = []
# #     wh_ratio=[]
#     areas_whratio_achor = []
#     for _ in range(data.steps_per_epoch):
        
#         batch_img, y1, y2, y3 ,areas_wh_ratio_achor = next(data)
#         areas_whratio_achor.extend(areas_wh_ratio_achor)


# #         areas.extend(ori_areas)
# #         selected_anchors.extend(selected_anchors_)
# #         wh_ratio.extend(ori_wh_ratio)
#         #print('batch_img.shape: ',batch_img.shape,'y1.shape, y2.shape, y3.shape: ',y1.shape ,y2.shape, y3.shape)
# #         num+=1
# #         if num>15:
# #             asd
        
#     np.savetxt('{}gt_area.csv'.format(f_dir), areas_whratio_achor , delimiter=",",header='area,whratio,achor,label_id', fmt='%s')
    df = pd.read_csv('{}gt_area.csv'.format(f_dir))
    
    #bins = [i for i in range(0,2073600,100)]
    bins = [i for i in range(0,820000,5000)]
#     print('area bins: ',bins)
    plt.figure(figsize=(100, 5))
    area = pd.cut(df['area'], bins)
    area_diagram = sns.countplot(area)
    plt.savefig('{}area820000_5000.png'.format(f_dir))
    
    
    whratio_bins = [i/50 for i in range(0,200,5)]
#     print('whratio bins: ',whratio_bins)
    plt.figure(figsize=(15, 5))
    whratio = pd.cut(df['whratio'], whratio_bins)
    whratio_diagram = sns.countplot(whratio)
    plt.savefig('{}whratio.png'.format(f_dir))
    
    plt.figure(figsize=(15, 5))
    achor_diagram = sns.countplot(df['achor'])
    plt.savefig('{}achor.png'.format(f_dir))
    
    plt.figure(figsize=(15, 5))
    achor_diagram = sns.countplot(df['label_id'])
    plt.savefig('{}label_id.png'.format(f_dir))
    
    print(df.describe())
    
    return 0


if __name__ == "__main__":
    
    # Log.add_loss("###########")
    test_data_draw()
