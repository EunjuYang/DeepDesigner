"""
evaluator.py
Evaluator model using resnet

Writer      : EJYang
Last Update : 2017.12.29
"""
import tensorflow as tf
from layer import *
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import numpy as np
import csv
import random


def readyTrainingData(testCSV, csvList):
    '''
    prepare training data label list
    :param testCSV: csv file name
    :param csvList: list to store csv strings
    :return:
    '''
    f = open(testCSV, "r")
    reader = csv.reader(f)
    for line in reader:
        line = map(str.strip, line)
        csvList.append(line)


def prepareData(testCSV):
    f = open(testCSV,"r")
    reader = csv.reader(f)
    height, width = 256, 256

    for line in reader:

        # remove all blank in string list
        line = map(str.strip, line)
        #print line
        img0 = imread(line[0])
        img0 = imresize(img0,(height,width))
        img1 = imread(line[1]).astype(np.float32)
        img1 = imresize(img1,(height,width))
        imsave(line[0],img0)
        imsave(line[1],img1)

prepareData("deep_designer_objective.csv")
#prepareData("vehicle_img.csv")

def getTrainingData(labelList, batchSize, trainSize, IsValidate=False):
    '''
    return Training Data
    :param labelList: list returned by readyTrainingData function
    :param batchSize: mini-batch size to use in training
    :param trainSize: size of training data
    :param IsValidate: boolean variable
    :return: training batch data
    '''
    # Prepare Input Data
    height, width = 256, 256
    IMG1 = np.zeros((batchSize, height, width, 3), dtype=np.float32)
    IMG2 = np.zeros((batchSize, height, width, 3), dtype=np.float32)
    labelset = np.zeros((batchSize, 1), dtype=np.float32)

    if IsValidate == False:
        idx = sorted(random.sample(xrange(0, trainSize), batchSize))
    else:
        idx = sorted(random.sample(xrange(0, batchSize), batchSize))

    for i in range(batchSize):
        # labelList
        line = labelList[idx[i]]
        # input images
        img0 = imread(line[0]).astype(np.float32)
        img1 = imread(line[1]).astype(np.float32)
        np.copyto(IMG1[i], img0)
        np.copyto(IMG2[i], img1)

        # label
        labelset[i] = float(line[2])

    # dimension of training data
    imgH = IMG1[0].shape[0]
    imgW = IMG1[0].shape[1]
    imgC = 3
    outputshape = 1
    IMG1 = np.array(IMG1)
    IMG2 = np.array(IMG2)

    return IMG1, IMG2, labelset, [imgH, imgW, imgC, outputshape]


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False,is_train=True):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: Use Xavier as default
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc layers.
    :return: The created variable
    '''

    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer, trainable=is_train)

    return new_variables


def batch_normalization_layer(input_layer, dimension, m, is_train=True):
    '''
    Helper function to do batch normalization
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: th 4D tensor after being normalized
    '''

    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta%d' % m, dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=is_train)
    gamma = tf.get_variable('gamma%d' % m, dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=is_train)
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)
    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride, m, is_train=True):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :retrun: 4D tensor. Y = Relu(batch_normalize(conv_layer, out_channel))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape, is_train=is_train)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel, m, is_train=is_train)

    output = tf.nn.relu(bn_layer)

    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride, m, is_train=True):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel, m, is_train=is_train)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape, is_train=is_train)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, output_channel, first_block=False, m=1, is_train=True):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor. get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to shrink the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalize and relu-ed.
    with tf.variable_scope('%d_conv1_in_block' % m):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel], is_train=is_train)
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride, m=m)

    with tf.variable_scope('%d_conv2_in_block' % m):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1, m=m, is_train=is_train)

    # When the channel of input layer and conv2 does not match
    # depth of input layers

    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def output_layer(input_layer, num_labels, m, is_train=True):
    '''
    :param input_layer: 2D tensor
    :param num_lagbels = int. How many output labels in total? (10 for cifar 10)
    :return: output layer Y = WX + B
    '''

    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights_%d' % m, shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0), is_train=is_train)
    fc_b = create_variables(name='fc_biase_%d' % m, shape=[num_labels], initializer=tf.zeros_initializer(), is_train=is_train)
    # fc_h = tf.nn.relu(tf.matmul(input_layer,fc_w)+fc_b)
    fc_h = (tf.matmul(input_layer, fc_w) + fc_b)
    return fc_h


def resNet(input_tensor_batch, n, reuse, m=1, is_train=True):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n + 1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, result=False. To build validation graph and share weights
    with train graph, reuse True
    :param m: i th resNet
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    with tf.variable_scope('%dconv0' % m, reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [4, 4, 3, 16], 1, m=m, is_train=is_train)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('%dconv1_%d' % (m, i), reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block=True, m=m, is_train=is_train)
            else:
                conv1 = residual_block(layers[-1], 16, m=m, is_train=is_train)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('%dconv2_%d' % (m, i), reuse=reuse):
            conv2 = residual_block(layers[-1], 32, is_train=is_train)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('%dconv3_%d' % (m, i), reuse=reuse):
            conv3 = residual_block(layers[-1], 64, is_train=is_train)
            layers.append(conv3)

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel, m, is_train=is_train)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 1, m=m, is_train=is_train)
        layers.append(output)

    return layers[-1]



def evaluator(batchSize, shape, is_train=True):
    '''
    evaluator model
    :param batchSize: batch size
    :param shape: shape of layers
    :return: placeholders I1, I2, Y and layers: [I1, I2, Y, flatten, y, loss]
    '''

    I1 = tf.placeholder(tf.float32, [batchSize, shape[0], shape[1], shape[2]], name=None)
    I2 = tf.placeholder(tf.float32, [batchSize, shape[0], shape[1], shape[2]], name=None)
    Y = tf.placeholder(tf.float32, [batchSize, shape[3]], name=None)

    # generate layer object
    resOut1 = resNet(I1, 1, reuse=False, m=1)
    resOut2 = resNet(I2, 1, reuse=False, m=2)

    flatten = tf.concat([resOut1, resOut2], 1)
    fc1 = (fc(input=flatten, device=0, nOut=256, isClient=True, isTensor=False, mean=0, stddev=0.01)).TFoperation()
    fc2 = (fc(input=fc1, device=0, nOut=256, isClient=True, isTensor=False, mean=0, stddev=0.01)).TFoperation()
    fc3 = (fc(input=fc2, device=0, nOut=1, isClient=True, isTensor=False, mean=1, stddev=0.1)).TFoperation()
    cost = tf.losses.mean_squared_error(labels=Y, predictions=fc3)

    return I1, I2, Y, flatten, fc3, cost

