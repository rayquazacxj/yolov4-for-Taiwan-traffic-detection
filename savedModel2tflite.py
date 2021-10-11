import tensorflow as tf
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--savedModel_dir',  type=str, default='../model_effi0_320_mosaic/savedModelv2')
parser.add_argument('--tflite_name',  type=str, default='../model_effi0_320_mosaic/savedModelv2.tflite')
args = parser.parse_args()
# Convert the model.

#tf.compat.v1.config.set_soft_device_placement(enabled=True)

# with tf.compat.v1.Session(config=config) as sess:
#     meta_graph_def = tf.compat.v1.saved_model.loader.load(sess,
#        [tf.compat.v1.saved_model.tag_constants.SERVING],"/work/fghuio4000/competition/model_effi0_320_mosaic/savedModelv2")  
    
#     #print(meta_graph_def)
#     signature = meta_graph_def.signature_def
#     print(signature['serving_default'].inputs)
#     print(signature['serving_default'].outputs)
    
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(args.savedModel_dir)
tflite_model = converter.convert()

# Save the model.
with open(args.tflite_name, 'wb') as f:
    f.write(tflite_model)
    
print('savedModel to tflite done!!')


