import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from datetime import datetime


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph=tf.get_default_graph()
    input_image=graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob=graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    layer3=graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4=graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7=graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    print("load_vgg-input_image:", input_image)
    #input_image.set_shape((None,224,224,3))
    print("load_vgg-input_image:", input_image)

    return input_image, keep_prob, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)

# encoder/decoder pipeline:

# vgg_layer3_out
# vgg_layer4_out
# vgg_layer7_out: input, final frozen vgg layer
# num_classes:2 classes ( i.e. binary/2 classes: road or !road )

# tf.layers.conv2d( inputs, filters, kernel_size, strides=(1, 1), padding='valid', ...
# kernel_size:
#   A tuple or list of 2 positive integers specifying the spatial dimensions of the filters.
#   Can be a single integer to specify the same value for all spatial dimensions.
# strides:
#   A tuple or list of 2 positive integers specifying the strides of the convolution.
#   Can be a single integer to specify the same value for all spatial dimensions.

REGULARIZEDLAYERS=False
STANDARDDEVIATION=0.01

def convT(suffix, input, numberOfChannels, filterSize, strideSize):
    convolutionName="convT_"+str(suffix)
    if REGULARIZEDLAYERS:
        return tf.layers.conv2d_transpose(input, numberOfChannels, filterSize, strideSize, padding="same",
            kernel_initializer= tf.random_normal_initializer(stddev=STANDARDDEVIATION), name=convolutionName,
            kernal_regularizer=tf.contrib.layers.l2_regularizer(1.e-3))
    else:
        return tf.layers.conv2d_transpose(input, numberOfChannels, filterSize, strideSize, padding="same",
            kernel_initializer= tf.random_normal_initializer(stddev=STANDARDDEVIATION), name=convolutionName)

def conv(suffix, input, numberOfChannels, filterSize, strideSize):
    convolutionName="conv_"+str(suffix)
    if REGULARIZEDLAYERS:
        return tf.layers.conv2d(input, numberOfChannels, filterSize, strideSize, padding="same",
            kernel_initializer= tf.random_normal_initializer(stddev=STANDARDDEVIATION), name=convolutionName,
            kernal_regularizer=tf.contrib.layers.l2_regularizer(1.e-3))
    else:
        return tf.layers.conv2d(input, numberOfChannels, filterSize, strideSize, padding="same",
            kernel_initializer= tf.random_normal_initializer(stddev=STANDARDDEVIATION), name=convolutionName)

def reduceChannels(suffix, input, numberOfChannels):
    print("reduceChannels-input:", input)
    return conv(suffix, input, numberOfChannels, 1, 1)

def increaseXY(suffix, input, ratio):
    print("increaseXY-filterSize:", ratio, ", input:", input)
    numberOfChannels=input.shape[3]
    return convT(suffix, input, numberOfChannels, ratio, ratio)

otherTensorNames=[
"layer7_out:0",
"conv_5x18x512/Conv2D:0", "conv_5x18x512/kernel:0",
"convT_10x36x512/conv2d_transpose:0", "convT_10x36x512/kernel:0",
"add10x36x512:0",
"conv_10x36x256/Conv2D:0", "conv_10x36x256/kernel:0",
"convT_20x72x256/conv2d_transpose:0", "convT_20x72x256/kernel:0",
"add20x72x256:0",
"conv_20x72x128/Conv2D:0", "conv_20x72x128/kernel:0",
"convT_40x144x128/conv2d_transpose:0", "convT_40x144x128/kernel:0",
"conv_40x144x64/Conv2D:0", "conv_40x144x64/kernel:0",
"convT_80x288x64/conv2d_transpose:0", "convT_80x288x64/kernel:0",
"conv_80x288x32/Conv2D:0", "conv_80x288x32/kernel:0",
"convT_160x576x32/conv2d_transpose:0", "convT_160x576x32/kernel:0",
"conv_160x576x2/Conv2D:0", "conv_160x576x2/kernel:0",

]

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # the FINAL convolutional transpose layer will be 4-dimensional: (batch_size, original_height, original_width, num_classes)

    # TODO: Implement function
    # ?,5,18,4096 -> ?,5,18,512
    channels512=reduceChannels("5x18x512", vgg_layer7_out, 512)
    print ("layers-channels512:", channels512) 
    # ?,5,18,512 -> 10x36x512
    xy10x36=increaseXY("10x36x512", channels512, 2)
    # add layer 4
    add10x36x512=tf.add(xy10x36, vgg_layer4_out, name="add10x36x512")
    # 10x36x512 -> 10x36x256
    channels256=reduceChannels("10x36x256", add10x36x512, 256)
    # 10x36x256 -> 20x72x256
    xy20x72=increaseXY("20x72x256", channels256, 2)
    # add layer 3
    add20x72x256=tf.add(xy20x72, vgg_layer3_out, name="add20x72x256")
    # 20x72x256 -> 20x72x128
    channels128=reduceChannels("20x72x128", add20x72x256, 128)
    # 20x72x128 -> 40x144x128
    xy40x144=increaseXY("40x144x128", channels128, 2)
    # 40x144x128 -> 40x144x64
    channels64=reduceChannels("40x144x64", xy40x144, 64)
    # 40x144x64 -> 80x288x64
    xy80x288=increaseXY("80x288x64", channels64, 2)
    # 80x288x64 -> 80x288x32
    channels32=reduceChannels("80x288x32", xy80x288, 32)
    # 80x288x32 -> 160x576x32
    xy160x576=increaseXY("160x576x32", channels32, 2)
    # 160x576x32 -> 160x576x2
    only2Channels=reduceChannels("160x576x2", xy160x576, num_classes)
    print ("layers-only2Channels:", only2Channels)

    return only2Channels

tests.test_layers(layers)

# FCN-8 — Classification and Loss:

# nn_last_layer:
# correct_label:
# learning_rate:
# num_classes:
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # tf.reshape: 4d to 2d: height/classes x width/pixels
    # 
    return None, None, None
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass
#tests.test_train_nn(train_nn)

vggTensorNames=[
"keep_prob:0", "image_input:0",
"conv1_1/Conv2D:0", "conv1_1/filter:0",
    "conv1_2/Conv2D:0", "conv1_2/filter:0",
    "pool1:0",
"conv2_1/Conv2D:0", "conv2_1/filter:0", 
    "conv2_2/Conv2D:0", "conv2_2/filter:0",
    "pool2:0",
"conv3_1/Conv2D:0", "conv3_1/filter:0",
    "conv3_2/Conv2D:0", "conv3_2/filter:0",
    "conv3_3/Conv2D:0", "conv3_3/filter:0",
    "pool3:0",
"layer3_out:0",
"conv4_1/Conv2D:0", "conv4_1/filter:0",
    "conv4_2/Conv2D:0", "conv4_2/filter:0",
    "conv4_3/Conv2D:0", "conv4_3/filter:0",
    "pool4:0",
"layer4_out:0",
"conv5_1/Conv2D:0", "conv5_1/filter:0",
    "conv5_2/Conv2D:0", "conv5_2/filter:0",
    "conv5_3/Conv2D:0", "conv5_3/filter:0",
    "pool5:0",
"fc6/Conv2D:0", "dropout/mul:0",
"fc7/Conv2D:0", "dropout_1/mul:0",
"layer7_out:0",
]

def run():
    num_classes = 2
    image_shape = (160, 576) # underlying images shape is 1241x375, resized in: helper.gen_batch_function
    data_dir = './data' # this is right
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg') # ./data/vgg
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape) # ./data/vgg/data_road/training, 170x576

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3, layer4, layer7=load_vgg(sess, vgg_path)
        nn_layers = layers(layer3, layer4, layer7, num_classes) # num_classes==2


        batch_size = 5

        helper.saveGraph(runs_dir, sess.graph)
        print("get_batches_fn:", type(get_batches_fn(batch_size)))
        image, label=get_batches_fn(batch_size)
        print ("run-image[0].type:",type(image[0]), ", shape:", image[0].shape)
        #reshapeLabel=label[0].reshape((-1,num_classes))
        reshapeLabel=label[0][:,:,:,0:2]
        print ("run-label[0].type:",type(label[0]), ", shape:", label[0].shape, ", reshapeLabel.shape:", reshapeLabel.shape)
        helper.showTensorSizes({input_image: image[0], keep_prob: 0.5}, sess, otherTensorNames)

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
