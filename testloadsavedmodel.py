import tensorflow as tf
import argparse

#https://blog.csdn.net/zmlovelx/article/details/100511406

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_prefix',  type=str, default='../model_yolov4_mosaicv2/voc-333720')
parser.add_argument('--export_dir',  type=str, default='../model_yolov4_mosaicv2/savedModel333kv3')
args = parser.parse_args()

# graph = tf.Graph()
config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
# with tf.compat.v1.Session(graph=graph, config=config) as sess:
    
with tf.compat.v1.Session(config=config) as sess:
    meta_graph_def = tf.compat.v1.saved_model.loader.load(sess,
       [tf.compat.v1.saved_model.tag_constants.SERVING],"/work/fghuio4000/competition/model_effi0_320_mosaic/savedModelv2")  
    
    #print(meta_graph_def)
    signature = meta_graph_def.signature_def
    print(signature['serving_default'].inputs['input'].name)



