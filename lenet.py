import tensorflow as tf 
from tensorflow.contrib.layers import flatten 
import numpy as np
import random 
import matplotlib.puplot as plt 
%matplotlib inline 
from sklearn.utils import shuffle 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",reshape = False)
X_train,y_train = mnist.train.images , mnist.train.labels
X_test,y_test = mnist.test.images , mnist.test.labels
X_validation , y_validation = mnist.validation.images , mnist.validation.labels


assert(len(X_train)) == len(y_train
assert(len(X_validation)) == len(y_validation)
assert(len(X_test)) == len(y_test)

X_train = np.pad(X_train ,((0,0),(2,2),(2,2),(0,0)),'constant')
X_validation = np.pad(X_validation ,((0,0),(2,2),(2,2),(0,0)),'constant')
X_test = np.pad(X_test,((0,0),(2,2),(2,2),(0,0)),'constant')


X_train , y_train = shuffle(X_train,y_train)


epochs = 10 
BATCH_SIZE = 128 


# *COMPLETE PIPELINE* 
# I/P -> W1*x+b -> ReLU -> AvgPool -> W2*x+b2 ->ReLU ->AvgPool ->flatten ->W3*x+b3 ->ReLU-> W4*x + b4 ->ReLU ->W5*x + b5->softmax -> crossentropy 



def LeNet(x):

    mu = 0
    sigma = 0.1

    # Weight Initialization
    # w ~ N(0, σ²)
    # Shape = (Filter Height, Filter Width, Input Channels, Number of Filters)

    wc1 = tf.Variable(
        tf.truncated_normal(
            shape=(5, 5, 1, 6),
            mean=mu,
            stddev=sigma
        )
    )

    # Bias Initialization
    # bm = 0

    bc1 = tf.Variable(tf.zeros(6))

    # Convolution
    #
    # y(i,j,m) =
    # Σ Σ Σ x(i+u,j+v,c) · w(u,v,c,m)
    #
    # u = 0→4
    # v = 0→4
    # c = 1

    conv1 = tf.nn.conv2d(
        x,
        wc1,
        strides=[1, 1, 1, 1],
        padding='VALID'
    )

    # Bias Addition
    #
    # z(i,j,m) = y(i,j,m) + bm

    conv1 = tf.nn.bias_add(conv1, bc1)

    # ReLU Activation
    #
    # a(i,j,m) = max(0, z(i,j,m))

    conv1 = tf.nn.relu(conv1)

    # Output Shape
    # 28 × 28 × 6



    # Average Pooling
    #
    # P(i,j)
    # = (1/4)
    # Σ Σ a(2i+u,2j+v)
    #
    # u = 0→1
    # v = 0→1

    pool1 = tf.nn.avg_pool(
        conv1,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='VALID'
    )

    # Output Shape
    # 14 × 14 × 6


    wc2 = tf.Variable(
        tf.truncated_normal(
            shape=(5, 5, 6, 16),
            mean=mu,
            stddev=sigma
        )
    )

    bc2 = tf.Variable(tf.zeros(16))

    # Convolution
    #
    # y(i,j,m)
    # =
    # Σ Σ Σ x(i+u,j+v,c) · w(u,v,c,m)
    #
    # u = 0→4
    # v = 0→4
    # c = 1→6

    conv2 = tf.nn.conv2d(
        pool1,
        wc2,
        strides=[1, 1, 1, 1],
        padding='VALID'
    )

    # Bias Addition
    #
    # z(i,j,m) = y(i,j,m) + bm

    conv2 = tf.nn.bias_add(conv2, bc2)

    # ReLU
    #
    # a(i,j,m)=max(0,z(i,j,m))

    conv2 = tf.nn.relu(conv2)

    # Output Shape
    # 10 × 10 × 16


    pool2 = tf.nn.avg_pool(
        conv2,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='VALID'
    )

    # Output Shape
    # 5 × 5 × 16


    # 5 × 5 × 16 → 400

    flat = tf.reshape(pool2, [-1, 5 * 5 * 16])



    wf1 = tf.Variable(
        tf.truncated_normal(
            shape=(400, 120),
            mean=mu,
            stddev=sigma
        )
    )

    bf1 = tf.Variable(tf.zeros(120))

    # Matrix Multiplication
    #
    # z = Wx + b
    #
    # zi = Σ Wijxj + bi

    fc1 = tf.matmul(flat, wf1) + bf1

    # ReLU
    #
    # ai=max(0,zi)

    fc1 = tf.nn.relu(fc1)

    # Output Shape
    # 120



    wf2 = tf.Variable(
        tf.truncated_normal(
            shape=(120, 84),
            mean=mu,
            stddev=sigma
        )
    )

    bf2 = tf.Variable(tf.zeros(84))

    # z = Wx + b

    fc2 = tf.matmul(fc1, wf2) + bf2

    # ai=max(0,zi)

    fc2 = tf.nn.relu(fc2)

    # Output Shape
    # 84



    wf3 = tf.Variable(
        tf.truncated_normal(
            shape=(84, 10),
            mean=mu,
            stddev=sigma
        )
    )

    bf3 = tf.Variable(tf.zeros(10))

    # Output Logits
    #
    # o = Wx + b

    logits = tf.matmul(fc2, wf3) + bf3

    # Output Shape
    # 10 (Digits 0-9)

    return logits
















