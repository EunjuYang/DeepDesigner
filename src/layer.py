#-*- coding: utf-8 -*-
"""
    layer.py FROM DLMDL2 (https://github.com/EunjooYang/DLMDL2)
    Contributor : Eunjoo Yang (yejyang@kaist.ac.kr), Taewoo Kim
    Last Update : 2017.06.09
    This file is written to define layer classes
"""
import numpy as np
import tensorflow as tf
#from apiGenerator import *

class layer:

    parameters = []
    data_format = "NHWC"
    layerName = "layer"
    BatchSize = 128

    def __init__(self, name, device, input=None, output=None):
        """
        Constructor of the layer class
        :param name: string with name of this layer
        :param input: string with name of the input layer for this layer object
        :param output: string with name of the output layer for this layer object
        :param device: string with name of the device (e.g., "dev0") which computes the computation for this layer object
        """
        self.name = name
        self.device = device
        self.input = input
        self.output = output

    def createLayer(self):
        return "%s = %s(%s)" % (self.name, layer.layerName, self.paramString)

    def getTensorCode(self):
        """
        getTensorCode method
        :return: ** sth can be interpreted to tensorflow API using protocol buffer
        """
        None

    def getName(self):
        """
        return name of this layer
        :return: self.name
        """
        return self.name

    def setInput(self, input):
        """
        set Input for this layer object
        :param input: string of the input layer for this layer object
        :return: None
        """
        self.input = input

    def setDataFormat(self, dataformat):
        layer.data_format = dataformat


class conv(layer):

    # count variable which count total number of convolution layer
    count = 0

    # DLMDL TF API
    layerName = "conv"

    def __init__(self, input, device, nIn, nOut, kH, kW, dH, dW, padType, mean=0, stdev=1e-1):
        """
        constructor for convolution layer
        :param input: string for input layer name
        :param output: string for output layer name
        :param device: string for the device name
        :param nIn: number of input channel
        :param nOut: number of output channel
        :param kH, kW: convolution filter height and width
        :param dH, dW: stride height and width
        :param padType: Type of padding algorithm to use. 'SAME' or 'VALID'
        """

        conv.count += 1
        self.name = "conv%d" % conv.count
        self.device = device
        self.input = input
        self.nIn = int(nIn)
        self.nOut = int(nOut)
        self.kH = int(kH)
        self.kW = int(kW)
        self.dH = int(dH)
        self.dW = int(dW)
        self.padType = padType
        self.mean = mean
        self.stdev = stdev
        if layer.data_format == 'NCHW':
            self.strides = [1, 1, self.dH, self.dW]
        else:
            self.strides = [1, self.dH, self.dW, 1]
        self.paramString = "input=%s, nIn=%d, nOut=%d, kH=%d, kW=%d, dH=%d, dW=%d, padType=%s" % (self.input, self.nIn, self.nOut, self.kH, self.kW, self.dH, self.dW, self.padType)


    #def __init__(self):
    #    None

    def TFoperation(self):
        with tf.name_scope(self.name) as scope:
            kernel = tf.Variable(tf.truncated_normal([self.kH, self.kW, self.nIn, self.nOut], dtype=tf.float32, mean=self.mean, stddev=self.stdev, name='weights'))
            conv = tf.nn.conv2d(self.input, kernel, self.strides, padding=self.padType, data_format=layer.data_format)
            biases = tf.Variable(tf.constant(0.0, shape=[self.nOut], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.reshape(tf.nn.bias_add(conv, biases, data_format=layer.data_format), conv.get_shape())
            conv_ = tf.nn.relu(bias, name=scope)
            return conv_, kernel


    def Caffeoperariton(self):
        '''
        Caffe conv layer description
        '''
        return

    def setInput(self, input):
        """
        setInput
        :param input: string for input layer name
        :return: None
        """
        self.input = input

    def createLayer(self):

        return "%s = %s(%s)" % (self.name, conv.layerName, self.paramString)

    def getName(self):
        """
        return name of this layer
        :return: self.name
        """
        return self.name

    def setInput(self, input):
        """
        set Input for this layer object
        :param input: string of the input layer for this layer object
        :return: None
        """
        self.input = input

class mpool(layer):

    count = 0
    layerName = "mpool"

    def __init__(self, input, device, kH, kW, dH, dW, padType):
        """
        constructor for max_pooling layer
        :param input: string for input layer name
        :param device: string for the device name
        :param kH, kW: sliding window height and width
        :param dH, dW: stride height and width
        :param padType: Type of padding algorithm to use. 'SAME' or 'VALID'
        """
        self.name = "mpool%d" % mpool.count
        mpool.count += 1
        self.device = device
        self.input = input
        self.kH = int(kH)
        self.kW = int(kW)
        self.dH = int(dH)
        self.dW = int(dW)
        self.padType = padType
        self.kernel = [1, self.kH, self.kW, 1]
        if layer.data_format == 'NCHW':
            self.strides = [1, 1, self.dH, self.dW]
        else:
            self.strides = [1, self.dH, self.dW, 1]
        self.paramString = "value=%s, kH=%d, kW=%d, dH=%d, dW=%d, padType=%s" % (self.input, self.kH, self.kW, self.dH, self.dW, self.padType)


    def TFoperation(self):
        mpool_ = tf.nn.max_pool(self.input, self.kernel, self.strides, self.padType, data_format=layer.data_format)
        return mpool_

    def Caffeoperariton(self):
        '''
        Caffe conv layer description
        '''
        return

    def setInput(self, input):
        """
        setInput
        :param input: string for input layer name
        :return: None
        """
        self.input = input

    def createLayer(self):

        return "%s = %s(%s)" % (self.name, mpool.layerName, self.paramString)

    def getName(self):
        """
        return name of this layer
        :return: self.name
        """
        return self.name

    def setInput(self, input):
        """
        set Input for this layer object
        :param input: string of the input layer for this layer object
        :return: None
        """
        self.input = input



class avgpool(layer):

    count = 0
    layerName = "avgpool"

    def __init__(self, input, device, kH, kW, dH, dW, padType):
        """
        constructor for avg_pooling layer
        :param input: string for input layer name
        :param device: string for the device name
        :param kH, kW: sliding window height and width
        :param dH, dW: stride height and width
        :param padType: Type of padding algorithm to use. 'SAME' or 'VALID'
        """
        self.name = "avgpool%d" % avgpool.count
        mpool.count += 1
        self.device = device
        exec("self.input = %s" % input)
        self.kH = int(kH)
        self.kW = int(kW)
        self.dH = int(dH)
        self.dW = int(dW)
        self.padType = padType
        self.kernel = [1, self.kH, self.kW, 1]
        if layer.data_format == 'NCHW':
            self.strides = [1, 1, self.dH, self.dW]
        else:
            self.strides = [1, self.dH, self.dW, 1]
        self.paramString = "input=%s, kH=%d, kW=%d, dH=%d, dW=%d, padType=%s" % (self.input, self.kH, self.kW, self.dH, self.dW, self.padType)

    def TFoperation(self):
        avgpool_ = tf.nn.avg_pool(self.input, self.kernel, self.strides, self.padType, data_format=layer.data_format)
        return avgpool_

    def Caffeoperariton(self):
        '''
        Caffe conv layer description
        '''
        return

    def setInput(self, input):
        """
        setInput
        :param input: string for input layer name
        :return: None
        """
        self.input = input

    def createLayer(self):

        return "%s = %s(%s)" % (self.name, avgpool.layerName, self.paramString)

    def getName(self):
        """
        return name of this layer
        :return: self.name
        """
        return self.name

    def setInput(self, input):
        """
        set Input for this layer object
        :param input: string of the input layer for this layer object
        :return: None
        """
        self.input = input

class fc(layer):

    count = 0
    layerName = "fc"

    def __init__(self, input, device, nOut, isClient=True, isTensor=True, mean = 0, stddev = 1e-2):
        """
        constructor for Fully-connected layer
        :param input: string for input layer name
        :param device: string for the device name
        :param nIn: number of input layer
        :param nOut: number of output layer
        """
        self.name = "fc%d" % fc.count
        fc.count += 1
        self.device = device
        if isClient:
            if isTensor:
                self.input = tf.reshape(input,[layer.BatchSize,-1])
                self.nIn = int(self.input.shape[1])
            else:
                self.input = input
                self.nIn = int(input.shape[1])
        else:
            self.input = input
        self.nOut = int(nOut)
        self.mean = mean
        self.stddev = stddev
        self.paramString = "input=%s, nIn=%d, nOut=%d" % (self.input, self.nIn, self.nOut)

    def TFoperation(self):
        with tf.name_scope(self.name) as scope:
            kernel = tf.Variable(tf.truncated_normal([self.nIn, self.nOut], dtype=tf.float32,mean=self.mean, stddev=self.stddev), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[self.nOut], dtype=tf.float32), trainable=True, name='biases')
            fc_= tf.nn.relu_layer(self.input, kernel, biases)
            return fc_

    def Caffeoperariton(self):
        '''
        Caffe conv layer description
        '''
        return

    def setInput(self, input):
        """
        setInput
        :param input: string for input layer name
        :return: None
        """
        self.input = input

    def createLayer(self):

        return "%s = %s(%s)" % (self.name, fc.layerName, self.paramString)

    def getTensorCode(self):
        """
        :return:
        """
        None


    def getName(self):
        """
        return name of this layer
        :return: self.name
        """
        return self.name

    def setInput(self, input):
        """
        set Input for this layer object
        :param input: string of the input layer for this layer object
        :return: None
        """
        self.input = input

class loss(layer):

    count = 0
    layerName = "loss"

    def __init__(self, device, logits, labels):
        """
        constructor for loss layer
        :param logits: predicted label of image
        :param labels: true label of image
        """
        self.name = "loss%d" % loss.count
        loss.count += 1
        self.device = device
        self.logits = logits
        self.labels = labels
        self.paramString = "logits=%s, labels=%s" % (self.logits, self.labels)

    def TFoperation(self):
        """
        batch_size = layer.BatchSize
        lables = tf.expand_dims(self.labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concated = tf.concat(1, [indices, lables])
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 1000]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=onehot_labels, name='xentropy')
        loss_ = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        """
        softmax_cross_engropy_with_logits = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.labels,name=None)
        loss_ = tf.reduce_mean(softmax_cross_engropy_with_logits)
        return loss_


    def Caffeoperariton(self):
        '''
        Caffe conv layer description
        '''
        return

    def setInput(self, input):
        """
        setInput
        :param input: string for input layer name
        :return: None
        """
        self.input = input

    def createLayer(self):

        return "%s = %s(%s)" % (self.name, loss.layerName, self.paramString)


    def getName(self):
        """
        return name of this layer
        :return: self.name
        """
        return self.name

    def setInput(self, input):
        """
        set Input for this layer object
        :param input: string of the input layer for this layer object
        :return: None
        """
        self.input = input

class norm(layer):

    count = 0
    layerName = "norm"

    def __init__(self, device, input, depth, bias, alpha, beta):
        """
        constructor for normalize layer
        :param input: string for input layer name
        :param depth: int for half-width of the 1-D normalization window. Default is 5
        :param bias: float for offset. Default is 1
        :param alpha: float for scale factor. Default is 1
        :param beta: float for exponent. Default is 0.5
        """
        self.name = "norm%d" % norm.count
        norm.count += 1
        self.device = device
        self.input = input
        self.depth = float(depth)
        self.bias = float(bias)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.paramString = "input=%s, depth=%f, bias=%f, alpha=%f, beta=%f" % (self.input, self.depth, self.bias, self.alpha, self.beta)

    def TFoperation(self):
        norm_ = tf.nn.lrn(self.input, depth_radius=self.depth, bias=self.bias, alpha=self.alpha, beta=self.beta)
        return norm_


    def Caffeoperariton(self):
        '''
        Caffe conv layer description
        '''
        return

    def setInput(self, input):
        """
        setInput
        :param input: string for input layer name
        :return: None
        """
        self.input = input

    def createLayer(self):

        return "%s = %s(%s)" % (self.name, norm.layerName, self.paramString)

    def getName(self):
        """
        return name of this layer
        :return: self.name
        """
        return self.name

    def setInput(self, input):
        """
        set Input for this layer object
        :param input: string of the input layer for this layer object
        :return: None
        """
        self.input = input


class adamopt(layer):

    count = 0
    layerName = "adamopt"

    def __init__(self, device, loss, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, train_iteration=1000, test_iteration=1, test_interval=50):
        """
        constructor for adam optimizer layer
        :param input: string for input layer name
        :param loss: loss to minimize
        :param learning_rate: learning rate
        :param depth: int for half-width of the 1-D normalization window. Default is 5
        :param bias: float for offset. Default is 1
        :param alpha: float for scale factor. Default is 1
        :param beta: float for exponent. Default is 0.5
        :param use_locking: bool. If True use lock for update operations. Default is False
        :train_iteration: int for total train iteration
        :test_iteration: int for total test iteration
        :test_interval: int for test interval each train iterations
        """
        self.name = "adamopt%d" % adamopt.count
        adamopt.count += 1
        self.device = device
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.use_locking = bool(use_locking)
        self.train_iteration = int(train_iteration)
        self.test_iteration = int(test_iteration)
        self.test_interval = int(test_interval)
        self.paramString = "loss=%s, learning_rate=%f, beta1=%f, beta2=%f, epsilon=%f, use_locking=%d, train_interval=%d, test_iteration=%d, test_interval=%d " % (self.loss, self.learning_rate, self.beta1, self.beta2, self.epsilon, self.use_locking, self.train_iteration, self.test_iteration, self.test_interval)
        self.global_step = tf.Variable(0,trainable=False)

    def TFoperation(self):
        with tf.device('/gpu:%d'%self.device):
            adamopt = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.use_locking)
            adamopt_ = adamopt.minimize(self.loss, self.global_step)
            return adamopt_


    def Caffeoperariton(self):
        '''
        Caffe conv layer description
        '''
        return

    def setInput(self, input):
        """
        setInput
        :param input: string for input layer name
        :return: None
        """
        self.input = input

    def createLayer(self):

        return "%s = %s(%s)" % (self.name, adamopt.layerName, self.paramString)

    def getTensorCode(self):
        """
        getTensorCode method
        :return: ** sth can be interpreted to tensorflow API using protocol buffer
        """
        TensorCode = []
        TensorCode += writeTF('정의돼야 할 함수') # 이 함수는 따로 다른 apiGenerator.py 에서 정의할 예정
        TensorCode += writeTF('정의돼야 할 함수') # 이 함수는 따로 다른 apiGenerator.py 에서 정의할 예정
        TensorCode += writeTF('정의돼야 할 함수') # 이 함수는 따로 다른 apiGenerator.py 에서 정의할 예정
        TensorCode += writeTF('정의돼야 할 함수') # 이 함수는 따로 다른 apiGenerator.py 에서 정의할 예정

        return TensorCode

    def getName(self):
        """
        return name of this layer
        :return: self.name
        """
        return self.name

    def setInput(self, input):
        """
        set Input for this layer object
        :param input: string of the input layer for this layer object
        :return: None
        """
        self.input = input


class adagradopt(layer):

    count = 0
    layerName = "adagradopt"

    def __init__(self, device, loss, learning_rate, initial_accumulator_value, use_locking, train_iteration, test_iteration, test_interval):
        """
        constructor for adagradient optimizer layer
        :param input: string for input layer name
        :param loss: loss to minimize
        :param learning_rate: learning rate
        :param initial_accumulator_value: float positive value. Starting value for the accumulators, must be positive
        :param use_locking: bool. If True use lock for update operations. Default is False
        :train_iteration: int for total train iteration
        :test_iteration: int for total test iteration
        :test_interval: int for test interval each train iterations
        """
        self.name = "adagradopt%d" % adagradopt.count
        adagradopt.count += 1
        self.device = device
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.initial_accumulator_value = float(initial_accumulator_value)
        self.use_locking = bool(use_locking)
        self.train_iteration = int(train_iteration)
        self.test_iteration = int(test_iteration)
        self.test_interval = int(test_interval)
        self.paramString = "loss=%s, learning_rate=%f, initial_accumulator_value=%f, use_locking=%d, train_interval=%d, test_iteration=%d, test_interval=%d " % (self.loss, self.learning_rate, self.initial_accumulator_value, self.use_locking, self.train_iteration, self.test_iteration, self.test_interval)

    def TFoperation(self):
        adagradopt = tf.train.AdagradOptimizer(self.learning_rate, self.initial_accumulator_value, self.use_locking)
        adagradopt_ = adagradopt.miniimze(self.loss)
        return adagradopt_


    def Caffeoperariton(self):
        '''
        Caffe conv layer description
        '''
        return

    def setInput(self, input):
        """
        setInput
        :param input: string for input layer name
        :return: None
        """
        self.input = input

    def createLayer(self):

        return "%s = %s(%s)" % (self.name, adagradopt.layerName, self.paramString)

    def getName(self):
        """
        return name of this layer
        :return: self.name
        """
        return self.name

    def setInput(self, input):
        """
        set Input for this layer object
        :param input: string of the input layer for this layer object
        :return: None
        """
        self.input = input
