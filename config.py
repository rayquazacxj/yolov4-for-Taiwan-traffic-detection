# coding:utf-8
# configuration file

# ############# Basic configuration. #############
size = width = height =608                    # image size
batch_size = 4
batch_size_tiny = 32
total_epoch = 60       # total epoch
save_per_epoch = 5        # per save_step save one model
data_debug = False       # load data in debug model (show pictures when loading images)
cls_normalizer = 1.0    # Loss coefficient of confidence
iou_normalizer = 0.07   # loss coefficient of ciou
iou_thresh = 0.5     # 
prob_thresh = 0.25      # 
score_thresh = 0.25     # 
val_score_thresh = 0.5      # 
val_iou_thresh = 0.213            # 
max_box = 100                # 
save_img = False #True             # save the result image when test the net

num_gpus = 2 #----------------------------------------------------------------

# ############# log #############
log_dir = './log'
log_name = 'log.txt'
loss_name = 'loss.txt'

# configure the leanring rate
lr_init = 2e-4                      # initial learning rate	# 0.00261
lr_lower =1e-6                  # minimum learning rate    
lr_type = 'cosine_decay_restart'#'piecewise'   # type of learning rate( 'exponential', 'piecewise', 'constant')
piecewise_boundaries = [3, 90, 100]   #  for piecewise
piecewise_values = [1e-4,1e-5, 1e-5, 1e-6]   # piecewise learning rate

# configure the optimizer
optimizer_type = 'sgd' # type of optimizer
momentum = 0.949          # 
weight_decay = 0.0005

# ############## train on VOC ##############
voc_root_dir = ["../ivslab_train"]  # root directory of voc dataset
voc_test_dir = "../ivslab_test_qualification/JPEGImages/All"#"/work/fghuio4000/competition/ivslab_test_public/ivslab_test_public/JPEGImages/All"                                                 # test pictures directory for VOC dataset
voc_save_dir = "../competition/val_iamges"                                                     # the folder to save result image for VOC dataset
voc_model_path = "../competition/model_v4_ema"#"./VOC_parallel_testing_4gpus"###   # the folder to save model for VOC dataset
pretrain_model_path = '../competition/yolov4_weight'

voc_model_name = "voc"                                          # the model name for VOC dataset
#voc_names = "./data/voc.names"                             # the names of voc dataset
voc_names = "./data/competition.names" 

voc_class_num = 4
#voc_anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326 #(for 416)
voc_anchors =28,36, 17,59 , 24,42, 46,58 ,28,98, 39,22, 65,105, 49,176 ,228,289 #(for 608)
voc_anchors_v4tiny =  30,61,  62,45,  59,119,  116,90,  156,198,  373,326 #(for 416)
