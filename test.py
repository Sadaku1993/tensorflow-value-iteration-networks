#coding:utf-8
import time
import numpy as np
import tensorflow as tf
from data import process_gridworld_data

# Data
tf.app.flags.DEFINE_string('input',           'data/gridworld_8.mat', 'Path to data')
tf.app.flags.DEFINE_integer('imsize',         8,                      'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs',         30,                     'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k',              10,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i',           2,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q',           10,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      12,                     'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 10,                     'Number of state inputs for each sample (real number, technically is k+1)')
tf.app.flags.DEFINE_boolean('untied_weights', False,                  'Untie weights of VI network')
# Misc.
tf.app.flags.DEFINE_integer('seed',           0,                      'Random seed for numpy')
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log',            False,                  'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir',          '/tmp/vintf/',          'Directory to store tensorboard summary')


# Seed
config = tf.app.flags.FLAGS
np.random.seed(config.seed)

# Placeholder
X = tf.placeholder(tf.float32, name="X", shape=[None, config.imsize,config.imsize, config.ch_i])
S1 = tf.placeholder(tf.int32, name="S1", shape=[None, config.statebatchsize])
S2 = tf.placeholder(tf.int32, name="S2", shape=[None, config.statebatchsize])
y = tf.placeholder(tf.int32, name="y", shape=[None])

init = tf.global_variables_initializer()

bias = tf.Variable(np.random.randn(1, 1, 1, config.ch_h) * 0.01, dtype=tf.float32)
w0 = tf.Variable(np.random.randn(3, 3, config.ch_i, config.ch_h) * 0.01, dtype=tf.float32)
w1 = tf.Variable(np.random.randn(1, 1, config.ch_h, 1) * 0.01, dtype=tf.float32)
w = tf.Variable(np.random.randn(3, 3, 1, config.ch_q) * 0.01, dtype=tf.float32)
v = tf.Variable(3, name='v')
with tf.Session() as sess:
    sess.run(init)

    print(X.shape)
    print(S1.shape)
    print(S2.shape)
    print(y.shape)

    # print(sess.run(w0))
    print(sess.run(v))

