import tensorflow as tf
import argparse

#https://blog.csdn.net/zmlovelx/article/details/100511406

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_prefix',  type=str, default='/work/fghuio4000/competition/model_effi0_320_mosaic/voc-556200')
parser.add_argument('--export_dir',  type=str, default='/work/fghuio4000/competition/model_effi0_320_mosaic/savedModelv2')
args = parser.parse_args()

graph = tf.Graph()
config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
with tf.compat.v1.Session(graph=graph, config=config) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(args.ckpt_prefix + '.meta')
    loader.restore(sess, args.ckpt_prefix )
    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(args.export_dir)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
    builder.save()
    
print('ckpt to savedModel done!!')


