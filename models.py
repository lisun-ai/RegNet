#!/usr/bin/env python3
"""
Main model
"""
import tensorflow as tf
from warp3D import *
from ops import *
from config import get_config
import numpy as np

config = get_config(is_train=True)

# Class for main registration network
class RegNet(object):
  def __init__(self, sess, config, name, is_train):
    self.sess = sess
    self.name = name
    self.is_train = is_train
    self.reuse = None
    self.base_map = tf.placeholder(tf.float32, [config.batch_size, config.im_size[0], config.im_size[1], config.im_size[2], 3])

    # moving / fixed images
    im_shape = [config.batch_size] + config.im_size + [1]
    self.x = tf.placeholder(tf.float32, im_shape)
    self.y = tf.placeholder(tf.float32, im_shape)
    self.x_mark = tf.placeholder(tf.int32, [config.batch_size, config.num_mark, 3])

    x_len, y_len, z_len = config.im_size

    # Network architecture
    x1 = conv3d(x, "conv11", 16, 3, 1, 
        "SAME", True, tf.nn.elu, True, self.is_train)
    x1 = conv3d(x1, "conv12", 16, 3, 1, 
        "SAME", True, tf.nn.elu, True, self.is_train)
    x1 = conv3d(x1, "conv13", 32, 3, 1, 
        "SAME", True, tf.nn.elu, True, self.is_train)
    x1 = conv3d(x1, "out1", 1, 3, 1, 
        "SAME", False, None, True, self.is_train)

    y1 = conv3d(y, "conv21", 16, 3, 1, 
        "SAME", True, tf.nn.elu, True, self.is_train)
    y1 = conv3d(y1, "conv22", 16, 3, 1, 
        "SAME", True, tf.nn.elu, True, self.is_train)
    y1 = conv3d(y1, "conv23", 32, 3, 1, 
        "SAME", True, tf.nn.elu, True, self.is_train)
    y1 = conv3d(y1, "out2", 1, 3, 1, 
        "SAME", False, None, True, self.is_train)

    xy = conv3d(tf.concat([x1, y1], 4), "conv31", 3, 3, 1, "SAME", True, tf.nn.elu, True, self.is_train)
    xy = conv3d(xy, "conv32", 16, 3, 1, 
        "SAME", True, tf.nn.elu, True, self.is_train)
    xy = conv3d(xy, "conv33", 16, 3, 1, 
        "SAME", True, tf.nn.elu, True, self.is_train)
    xy = conv3d(xy, "conv34", 16, 3, 1, 
        "SAME", True, tf.nn.elu, True, self.is_train)
    xy = conv3d(xy, "out3", 3, 3, 1, 
        "SAME", False, None, True, self.is_train)

    z, x_transformed, self.grid_x, self.grid_y, self.grid_z =\
      batch_warp3d(x, xy, self.x_mark, self.base_map)
    if self.reuse is None:
      self.var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      self.saver = tf.train.Saver(max_to_keep=500)
      self.reuse = True

    if self.is_train :
      self.y_mark = tf.placeholder(tf.int32, [config.batch_size, config.num_mark, 3])
      self.loss = l2loss(z, self.y)
      self.optim = tf.train.MomentumOptimizer(learning_rate=config.lr, momentum=0.9)
      self.train = self.optim.minimize(
         self.loss)

    self.sess.run(
      tf.global_variables_initializer())

  def fit(self, batch_x, batch_y, x_mark, y_mark, base_map):
    _, loss, grid_x, grid_y, grid_z = \
        self.sess.run([self.train, self.loss, self.grid_x, self.grid_y, self.grid_z], 
        {self.x:batch_x, self.y:batch_y, self.x_mark:x_mark, self.y_mark:y_mark, self.base_map:base_map})
    return loss, grid_x, grid_y, grid_z

  def deploy(self, dir_path, x, y):
    z = self.sess.run(self.z, {self.x:x, self.y:y})

  def save(self, dir_path, iter_num):
    self.saver.save(self.sess, dir_path+"/model-"+str(iter_num)+".ckpt")

  def restore(self, dir_path):
    self.saver.restore(self.sess, dir_path+"/model.ckpt")
