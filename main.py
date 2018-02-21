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
    print("global variables:\n", [n.name for n in tf.get_default_graph().as_graph_def().node])
    input_image=graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob=graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    layer3=graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4=graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7=graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
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

strides: A tuple or list of 2 positive integers specifying the strides of the convolution. Can be a single integer to specify the same value for all spatial dimensions.
REGULARIZEDLAYERS=False
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    if REGULARIZEDLAYERS:
        conv_1x1=tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding="same", kernal_regularizer=tf.contrib.layers.l2_regularizer(1.e-3))
        output=tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding="same", kernal_regularizer=tf.contrib.layers.l2_regularizer(1.e-3))
    else:
        conv_1x1=tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding="same", name="conv_1x1")
        upSample7=tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding="same",
            kernel_initializer= tf.random_normal_initializer(stddev=0.01), name="cov2dT_7")
        print ("layers-upSample7:", upSample7, " == ", vgg_layer7_out)
        conv_1x1_7=tf.layers.conv2d(upSample7, num_classes, 1, padding="same", name="conv_1x1_7")
        print ("layers-conv_1x1_7:", conv_1x1_7)
 
        upSample4=tf.layers.conv2d_transpose(conv_1x1_7, num_classes, 4, 2, padding="same",
            kernel_initializer= tf.random_normal_initializer(stddev=0.01), name="cov2dT_4")
        print ("layers-upSample4:", upSample4, " == ", vgg_layer4_out)
        conv_1x1_4=tf.layers.conv2d(upSample4, num_classes, 1, padding="same", name="conv_1x1_4")
        print ("layers-conv_1x1_4:", conv_1x1_4)

        upSample3=tf.layers.conv2d_transpose(conv_1x1_4, num_classes, 4, 2, padding="same",
            kernel_initializer= tf.random_normal_initializer(stddev=0.01), name="cov2dT_3")
        print ("layers-upSample3:", upSample3, " == ", vgg_layer3_out)
        conv_1x1_3=tf.layers.conv2d(upSample3, num_classes, 1, padding="same", name="conv_1x1_3")
        print ("layers-conv_1x1_3:", conv_1x1_3)    

    output=conv_1x1_3
    print ("layers-output:", output)

    return output

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


def run():
    num_classes = 2
    image_shape = (160, 576) # underlying images shape is 1241x375, resized in: helper.gen_batch_function
    data_dir = './data' # this is right
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir="/tmp"
    logdir="{}/tensorflow-{}/".format(root_logdir, now)


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
        tensorboard_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        tensorboard_writer.close()

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
