# coding:utf-8
# test on voc
import argparse
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import config_efficientlite_sh_ as config
from utils import tools
from src.YOLO_efficientlite_wdropblock import YOLO
import cv2
import numpy as np
from src import Log
import os
from os import path
from os import makedirs
import time
from src.Feature_parse_tf import get_predict_result

import pandas as pd

class_num = config.voc_class_num
# width = config.width
# height = config.height
#anchors =  np.asarray(config.voc_anchors).astype(np.float32).reshape([-1, 3, 2])
score_thresh = config.val_score_thresh
iou_thresh = config.val_iou_thresh
max_box = config.max_box
model_path = config.voc_model_path
name_file = config.voc_names
val_dir = config.voc_test_dir
#save_img = config.save_img
#save_dir = config.voc_save_dir
MOVING_AVERAGE_DECAY = 0.99

# output to csv
image_filename=[]
label_id=[]
x=[]
y=[]
w=[]
h=[]
confidence=[]

def read_img(img_name, width, height):
    img_ori = tools.read_img(img_name)
    if img_ori is None:
        return None, None
    ori_h, ori_w, _ = img_ori.shape
    img = cv2.resize(img_ori, (width, height))

    show_img = img
    
    img = img.astype(np.float32)
    img = img/255.0
    # [416, 416, 3] => [1, 416, 416, 3]
    img = np.expand_dims(img, 0)
    return img, ori_w, ori_h, img_ori, show_img

def write_img(img, name,save_dir):
    '''
    img: mat 
    name: saved name
    '''
    if not path.isdir(save_dir):
        Log.add_log("message: create floder'"+str(save_dir)+"'")
        os.mkdir(save_dir)
    img_name = path.join(save_dir, name)
    cv2.imwrite(img_name, img)
    return 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shape',  type=int, default=320)
    parser.add_argument('--model_path',  type=str, default="/work/fghuio4000/competition/model_effi0_416_mosaic_dropblock")
    parser.add_argument('--efficientlite_model',  type=str,default= 'efficientnet_lite0')
    parser.add_argument('--save_dir',  type=str,default= '/work/fghuio4000/competition/model_effi0_416_mosaic_dropblock/test')
    parser.add_argument('--save_img',  type=bool,default= True)
    #parser.add_argument('--total_epoch',  type=int, default= 30)
    #parser.add_argument('--optimizer_type', type=str, default= 'momentum')
    args = parser.parse_args()
    
    width = args.input_shape
    height = args.input_shape
    size = args.input_shape
    #total_epoch = args.total_epoch
    #optimizer_type= args.optimizer_type
    model_path =  args.model_path
    save_dir = args.save_dir
    save_img = args.save_img
    
    if args.input_shape==320:
        anchors =  np.asarray(config.voc_anchors_320).astype(np.float32).reshape([-1, 3, 2])
        print('args.input_shape==320')
    elif args.input_shape==416:
        anchors =  np.asarray(config.voc_anchors_416).astype(np.float32).reshape([-1, 3, 2])
        print('args.input_shape==416')
    elif args.input_shape==512:
        anchors =  np.asarray(config.voc_anchors_512).astype(np.float32).reshape([-1, 3, 2])
        print('args.input_shape==512')
    elif args.input_shape==608:
        anchors =  np.asarray(config.voc_anchors_608).astype(np.float32).reshape([-1, 3, 2])
        print('args.input_shape==608')
    elif args.input_shape==1088:
        anchors =  np.asarray(config.voc_anchors_1088).astype(np.float32).reshape([-1, 3, 2])
        print('args.input_shape==1088')    
    
    if not path.exists(save_dir):
        makedirs(save_dir)
        print('create model save folder: ',save_dir)
        
    yolo = YOLO()

    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, width, height, 3])
    feature_y1, feature_y2, feature_y3 = yolo.forward(inputs, class_num, model_name=args.efficientlite_model,isTrain=False)
    pre_boxes, pre_score, pre_label = get_predict_result(feature_y1, feature_y2, feature_y3,
                                                                                                anchors[2], anchors[1], anchors[0], 
                                                                                                width, height, class_num, 
                                                                                                score_thresh=0.005, 
                                                                                                iou_thresh=0.005,
                                                                                                max_box=500)

    init = tf.compat.v1.global_variables_initializer()

    
    
    
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    #saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        ckpt = tf.compat.v1.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            Log.add_log("message: load ckpt model:'"+str(ckpt.model_checkpoint_path)+"'")
        else:
            Log.add_log("message:can not find  ckpt model")
            # exit(1)
            # assert(0)
        
        # dictionary of name of corresponding id
        word_dict = tools.get_word_dict(name_file)
        # dictionary of per names
        color_table = tools.get_color_table(class_num)
        
        num=0
        for name in os.listdir(val_dir):
            num+=1
#             if num>15:
#                 asd
            
            img_name = path.join(val_dir, name)
            print('img_name: ',img_name)
            
            if not path.isfile(img_name):
                print("'%s' is not file" %img_name)
                continue

            img, nw, nh, img_ori, show_img = read_img(img_name, width, height)
            if img is None:
                Log.add_log("message:'"+str(img)+"' is None")
                continue

            start = time.perf_counter()
            
            boxes, score, label = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img})
            print('boxes, score, label: ',boxes, score, label) #boxes:[V, 4],float x_min, y_min, x_max, y_max
            for i in range(len(label)):
                image_filename.append(name)
                label_id.append(label[i]+1)
                
                x_min = max(boxes[i][0]*1920 , 0)
                y_min = max(boxes[i][1]*1080 , 0)
                x_max = min(boxes[i][2]*1920 , 1920)
                y_max = min(boxes[i][3]*1080 , 1080)
                ww = x_max - x_min
                hh = y_max - y_min
                
                x.append(x_min)
                y.append(y_min)
                w.append(ww)
                h.append(hh)
                confidence.append(score[i])
            
            end = time.perf_counter()
            print("%s\t, time:%f s" %(img_name, end-start))
           
            img_ori = tools.draw_img(img_ori, boxes, score, label, word_dict, color_table)
            #cv2.imwrite('test_compe.jpg',img_ori)
#             cv2.imshow('img', img_ori)
#             cv2.waitKey(0)
            
            if save_img:
                write_img(img_ori, name,save_dir )
            pass
        
        results = {'image_filename': image_filename,
                   'label_id': label_id,
                   'x':x,
                   'y':y,
                   'w':w,
                   'h':h,
                   'confidence':confidence
        }

        df = pd.DataFrame(results, columns= ['image_filename', 'label_id', 'x', 'y', 'w', 'h', 'confidence'])
        df = df.sort_index(by = ['image_filename'])  
        df.to_csv (r'{}/submission.csv'.format(save_dir), index = False, header=True)
        print('df:',df)

if __name__ == "__main__":
    Log.add_log("message: into val.main()")
    main()
