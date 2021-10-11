# coding:utf-8
# training on voc

#for parallel okk
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
from src.Data_voc import Data
from src.YOLO import YOLO
from os import path
from os import environ
import config
import time
import numpy as np
from src import Log
from src import Optimizer
from src import Learning_rate as Lr
from src.Loss import Loss
from src.parallel import average_gradients,assign_to_device

#environ['CUDA_VISIBLE_DEVICES'] = '0,1'
width = config.width
height = config.height
print('w:',width,' h: ',height)
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

#pretrain_model_path = config.pretrain_model_path
names_file = config.voc_names
data_debug = config.data_debug
model_name = config.voc_model_name
model_path = config.voc_model_path
total_epoch = config.total_epoch
save_per_epoch = config.save_per_epoch
voc_root_dir = config.voc_root_dir
print('voc_root_dir: ',voc_root_dir)



#  get current epoch 
def compute_curr_epoch(global_step, batch_size, imgs_num):
    '''
    global_step:current step
    batch_size:batch_size
    imgs_num: total images number
    '''
    epoch = global_step * batch_size / imgs_num
    return  tf.cast(epoch, tf.int32)


# for parallel--------------------------------------------------------------------
num_gpus = config.num_gpus


# def solve_cudnn_error():
#     #gpus = ConfigProto.experimental.list_physical_devices('GPU')
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         try:
#             # Currently, memory growth needs to be the same across GPUs
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#         except RuntimeError as e:
#             # Memory growth must be set before GPUs have been initialized
#             print(e)


#------------------------------------------------------------------------------------

# training
def backward():
    yolo = YOLO()
    tf.reset_default_graph() 
    # dataset
    print('train before data')
    data = Data(voc_root_dir, names_file, class_num, batch_size*num_gpus, 
                            anchors, is_tiny=False, size=size)
    imgs_ls = data.imgs_path
    labels_ls = data.labels_path
    print('train imgs_ls: ',imgs_ls[:5])
    print('train labels_ls: ',labels_ls[:5])
    # create dataset
    dataset = tf.data.Dataset.from_tensor_slices((imgs_ls, labels_ls))
    dataset = dataset.shuffle(len(imgs_ls))   # shuffle
    dataset = dataset.batch(batch_size=batch_size *num_gpus)    # for parallel
    dataset = dataset.map(
        lambda imgs_batch, xmls_batch: tf.py_func(
                data.load_tf_batch_data, 
                inp=[(imgs_batch, xmls_batch)],
                Tout=[tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=4*num_gpus
    )
    dataset = dataset.prefetch(20)
    # repeat
    dataset = dataset.repeat()
    # iterator
    iterator = dataset.make_initializable_iterator()
    inputs, y1_true, y2_true, y3_true = iterator.get_next()
    
    # set shape
    with tf.device('/cpu:0'):
        tower_grads = []
        reuse_vars = None
        #inputs.set_shape([batch_size  *num_gpus, None, None, None, 3])
        inputs.set_shape([None, None, None, 3])
        y1_true.set_shape([batch_size *num_gpus, None, None, 3, 5+class_num]) # 5: xywh,score
        y2_true.set_shape([batch_size *num_gpus, None, None, 3, 5+class_num])
        y3_true.set_shape([batch_size *num_gpus, None, None, 3, 5+class_num])
        
        
        #with tf.variable_scope('gpus') :
        for i in range(num_gpus):
            print('/gpu: ',i)
            with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                with tf.name_scope('GPU_%d' % i) as scope:  
                    
                    print('tf.get_variable_scope().reuse: ',tf.get_variable_scope().reuse)
                    print('tf.get_variable_scope().original_name_scope: ',tf.get_variable_scope().original_name_scope)
                    # Split data between GPUs
                    #_inputs =   inputs[i * batch_size: (i+1) * batch_size,None, None, None, 3]
                    _inputs =   inputs[i * batch_size: (i+1) * batch_size]
    #                 _y1_true = y1_true[i * batch_size: (i+1) * batch_size, None, None, 3, 5+class_num]
    #                 _y2_true = y2_true[i * batch_size: (i+1) * batch_size, None, None, 3, 5+class_num]
    #                 _y3_true = y3_true[i * batch_size: (i+1) * batch_size, None, None, 3, 5+class_num]
                    _y1_true = y1_true[i * batch_size: (i+1) * batch_size, ...]
                    _y2_true = y2_true[i * batch_size: (i+1) * batch_size, ...]
                    _y3_true = y3_true[i * batch_size: (i+1) * batch_size, ...]



                    # Because Dropout have different behavior at training and prediction time, we
                    # need to create 2 distinct computation graphs that share the same weights.

                    # Create a graph for training

                    feature_y1, feature_y2, feature_y3 = yolo.forward(_inputs, class_num, weight_decay=weight_decay, isTrain=True,reuse=reuse_vars)

                # Create another graph for testing that reuse the same weights


    #                 feature_y1_test, feature_y2_test, feature_y3_test = yolo.forward(_inputs, class_num, weight_decay=weight_decay, isTrain=True,reuse=True)

                    global_step = tf.Variable(0, trainable=False)

                    # loss value of yolov4
                    loss = Loss().yolo_loss([feature_y1, feature_y2, feature_y3], 
                                                                        [_y1_true, _y2_true, _y3_true], 
                                                                        [anchors[2], anchors[1], anchors[0]], 
                                                                        width, height, class_num,
                                                                        cls_normalizer=cls_normalizer,
                                                                        iou_normalizer=iou_normalizer,
                                                                        iou_thresh=iou_thresh, 
                                                                        prob_thresh=prob_thresh, 
                                                                        score_thresh=score_thresh)
                    l2_loss = tf.compat.v1.losses.get_regularization_loss() 

                    reuse_vars = True #
                    tf.get_variable_scope().reuse_variables()

                    epoch = compute_curr_epoch(global_step, batch_size, len(data.imgs_path))


                    lr = Lr.config_lr(lr_type, lr_init, lr_lower=lr_lower, \
                                                        piecewise_boundaries=piecewise_boundaries, \
                                                        piecewise_values=piecewise_values, epoch=epoch)
                    #optimizer = Optimizer.config_optimizer(optimizer_type, lr, momentum)
                    optimizer = Optimizer.config_optimizer('sgd', lr, momentum)

                    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS,scope = scope) # scope = scope add 2/10
                    #update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS) 
                    with tf.control_dependencies(update_ops): #can ensure run update_ops fisrt ,then do things within below code block 
                        gvs = optimizer.compute_gradients(loss+l2_loss)
                        clip_grad_var = [gv if gv[0] is None else[tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
                        #print('clip_grad_var: ',clip_grad_var)
    #                     train_step = optimizer.apply_gradients(clip_grad_var, global_step=global_step)


                    reuse_vars = True #
                    tower_grads.append(clip_grad_var) #
                    tf.get_variable_scope().reuse_variables()

        #print('tower_grads: ',tower_grads)
        
        tower_grads = average_gradients(tower_grads) #
        tower_clip_grad_var = [tower_gv if tower_gv[0] is None else[tf.clip_by_norm(tower_gv[0], 100.), tower_gv[1]] for tower_gv in tower_grads]
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            
            train_op = optimizer.apply_gradients(tower_clip_grad_var, global_step=global_step) #

    #solve_cudnn_error()
    
    # initialize
    config_ = ConfigProto(allow_soft_placement=True)
    config_.gpu_options.allow_growth = True
    init = tf.compat.v1.global_variables_initializer()

    saver = tf.compat.v1.train.Saver()
    
    with tf.compat.v1.Session(config = config_) as sess:
        sess.run(init)
        sess.run(iterator.initializer)
        step = 0
        
        
        ckpt = tf.compat.v1.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            #step = eval(step)
            step = 0
            Log.add_log("message: load ckpt model, global_step=" + str(step))
        else:
            Log.add_log("message:can not fint ckpt model")
        
        curr_epoch = step // data.steps_per_epoch
        print('curr_epoch: ',curr_epoch)
        
        while curr_epoch < total_epoch:
            for _ in range(data.steps_per_epoch):
                start = time.perf_counter()
                _, loss_, step, lr_ = sess.run([train_op, loss, global_step, lr])
                #_ , step, lr_ = sess.run([train_op,global_step, lr])
                end = time.perf_counter()
                
#                 if (loss_ > 1e3) and (step > 1e3):
#                     Log.add_log("error:loss exception, loss_value = "+str(loss_))
#                     ''' break the process or lower learning rate '''
#                     raise ValueError("error:loss exception, loss_value = "+str(loss_)+", please lower your learning rate")
#                     # lr = tf.math.maximum(tf.math.divide(lr, 10), config.lr_lower)

                if step % 5 == 2:
                    print("step: %6d, epoch: %3d, loss: %.5g\t, wh: %3d, lr:%.5g\t, time: %5f s"
                                %(step, curr_epoch, loss_, width, lr_, end-start))
#                     print("step: %6d, epoch: %3d, , wh: %3d, time: %5f s"
#                                 %(step, curr_epoch,  width, end-start))
                    Log.add_loss(str(step) + "\t" + str(loss_))

            curr_epoch += 1
            if curr_epoch % save_per_epoch == 0:
                # save ckpt model
                Log.add_log("message: save ckpt model, step=" + str(step) +", lr=" + str(lr_))
                #Log.add_log("message: save ckpt model, step=" + str(step))
                saver.save(sess, path.join(model_path, model_name), global_step=step)                    
                
        Log.add_log("message: save final ckpt model, step=" + str(step))
        saver.save(sess, path.join(model_path, model_name), global_step=step)

    return 0


if __name__ == "__main__":
    Log.add_log("message: into  VOC backward function")
    
    # Log.add_loss("###########")
    backward()
    tf.reset_default_graph()