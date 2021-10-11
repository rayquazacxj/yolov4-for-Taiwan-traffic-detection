# coding:utf-8
# configuration file

# ############# Basic configuration. #############
size = width = height = 608                     # image size
batch_size = 4
batch_size_tiny = 32
total_epoch = 40       # total epoch
save_per_epoch = 5        # per save_step save one model
data_debug = False       # load data in debug model (show pictures when loading images)
cls_normalizer = 1.0    # Loss coefficient of confidence
iou_normalizer = 0.07   # loss coefficient of ciou
iou_thresh = 0.5     # 
prob_thresh = 0.25      # 
score_thresh = 0.25     # 
val_score_thresh = 0.5      # 
val_iou_thresh = 0.213            # 
max_box = 60                # 
save_img = False #True             # save the result image when test the net

num_gpus = 4 #----------------------------------------------------------------

# ############# log #############
log_dir = './log'
log_name = 'log.txt'
loss_name = 'loss.txt'

# configure the leanring rate
lr_init = 2e-4                      # initial learning rate	# 0.00261
lr_lower =1e-6                  # minimum learning rate    
lr_type = 'piecewise'   # type of learning rate( 'exponential', 'piecewise', 'constant')
piecewise_boundaries = [3, 90, 100]   #  for piecewise
piecewise_values = [2e-4,1e-4, 1e-5, 1e-6]   # piecewise learning rate

# configure the optimizer
optimizer_type = 'momentum' # type of optimizer
momentum = 0.949          # 
weight_decay = 0.0005

# ############## train on VOC ##############
voc_root_dir = ["../ivslab_train"]  # root directory of voc dataset
voc_test_dir = "../ivslab_test_public/ivslab_test_public/JPEGImages/All"                                                 # test pictures directory for VOC dataset
voc_save_dir = "../val_iamges"                                                     # the folder to save result image for VOC dataset
voc_model_path = "../models_yolov4_608ema"#"./VOC_parallel_testing_4gpus"###   # the folder to save model for VOC dataset
pretrain_model_path = '../yolov4_weight'

voc_model_name = "voc"                                          # the model name for VOC dataset
#voc_names = "./data/voc.names"                             # the names of voc dataset
voc_names = "./data/competition.names" 

voc_class_num = 4
voc_anchors = 15,19, 23,44, 48,34, 44,90, 91,66, 86,174, 170,132, 228,289, 545,476
voc_anchors_320 = 8,10, 13,23, 25,18,  23,47,  48,35, 45,92, 89,69, 120,152, 286,250 #( for 320)
voc_anchors_416 = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326 #(for 416)
voc_anchors_512 = 12,16, 20,37, 41,28, 37,75, 76,55, 73,147, 143,111, 192,244, 460,401
voc_anchors_608 = 15,19, 23,44, 48,34, 44,90, 91,66, 86,174, 170,132, 228,289, 545,476
#voc_anchors_v4tiny =  30,61,  62,45,  59,119,  116,90,  156,198,  373,326 #(for 416)
