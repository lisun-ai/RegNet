#!/usr/bin/env python3
"""
Utilities
"""

import tensorflow as tf
import os
import numpy as np
from tensorflow.python.training import moving_averages
from keras import backend as K
BN_COLLECTION = tf.GraphKeys.UPDATE_OPS

def conv3d(x, name, dim, k, s, p, bn, af, reg, is_train):
  with tf.variable_scope(name):
    if reg:
      w = tf.get_variable('weight', [k, k, k, x.get_shape()[-1], dim],
        initializer=tf.truncated_normal_initializer(stddev=0.01),
        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5))
    else:
      w = tf.get_variable('weight', [k, k, k, x.get_shape()[-1], dim],
        initializer=tf.truncated_normal_initializer(stddev=0.01))

    x = tf.nn.conv3d(x, w, [1, s, s, s, 1], p)

    if bn:
      x = batch_norm(x, "bn", is_training=is_train)
    else:
      b = tf.get_variable('biases', shape=dim,
        initializer=tf.constant_initializer(0.))
      x=tf.nn.bias_add(x, b)

    if af:
      x = af(x)

  return x

def batch_norm(inputs, name, moving_decay=0.9, eps=1e-5, is_training=True):
  initializers = {'beta': tf.constant_initializer(0.0),
                             'gamma': tf.constant_initializer(1.0),
                             'moving_mean': tf.constant_initializer(0.0),
                             'moving_variance': tf.constant_initializer(1.0)}

  regularizers = {'beta': None, 'gamma': None}
  input_shape = inputs.shape

        # operates on all dims except the last dim
  params_shape = input_shape[-1:]
  axes = list(range(input_shape.ndims - 1))

        # create trainable variables and moving average variables
  beta = tf.get_variable(
            'beta',
            shape=params_shape,
            initializer=initializers['beta'],
            regularizer=regularizers['beta'],
            dtype=tf.float32, trainable=True)
  gamma = tf.get_variable(
            'gamma',
            shape=params_shape,
            initializer=initializers['gamma'],
            regularizer=regularizers['gamma'],
            dtype=tf.float32, trainable=True)

  collections = [tf.GraphKeys.GLOBAL_VARIABLES]
  moving_mean = tf.get_variable(
            'moving_mean',
            shape=params_shape,
            initializer=initializers['moving_mean'],
            dtype=tf.float32, trainable=False, collections=collections)
  moving_variance = tf.get_variable(
            'moving_variance',
            shape=params_shape,
            initializer=initializers['moving_variance'],
            dtype=tf.float32, trainable=False, collections=collections)
        # mean and var
  mean, variance = tf.nn.moments(inputs, axes)
  update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, moving_decay).op
  update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, moving_decay).op
  tf.add_to_collection(BN_COLLECTION, update_moving_mean)
  tf.add_to_collection(BN_COLLECTION, update_moving_variance)

  # call the normalisation function
  if is_training or use_local_stats:
    outputs = tf.nn.batch_normalization(
                inputs, mean, variance,
                beta, gamma, eps, name='batch_norm')
  else:
    outputs = tf.nn.batch_normalization(
                inputs, moving_mean, moving_variance,
                beta, gamma, eps, name='batch_norm')
  outputs.set_shape(inputs.get_shape())
  return outputs

def ncc(x, y):
  mean_x = tf.reduce_mean(x, [1,2,3], keep_dims=True)
  mean_y = tf.reduce_mean(y, [1,2,3], keep_dims=True)
  mean_x2 = tf.reduce_mean(tf.square(x), [1,2,3], keep_dims=True)
  mean_y2 = tf.reduce_mean(tf.square(y), [1,2,3], keep_dims=True)
  stddev_x = tf.reduce_sum(tf.sqrt(
    mean_x2 - tf.square(mean_x)), [1,2,3], keep_dims=True)
  stddev_y = tf.reduce_sum(tf.sqrt(
    mean_y2 - tf.square(mean_y)), [1,2,3], keep_dims=True)
  return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))

def mse(x, y):
  return tf.reduce_mean(tf.square(x - y))

def l2loss(prediction, ground_truth):
  residuals = tf.reduce_mean(tf.subtract(prediction, ground_truth))
  return tf.nn.l2_loss(residuals)
