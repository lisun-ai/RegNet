#!/usr/bin/env python3
"""
Config for training and testing of network
"""

class Config(object):
  pass

def get_config(is_train):
  config = Config()
  config.im_size = [200] * 3
  if is_train:
    config.batch_size = 1
    config.lr = 1e-3
    config.iteration = 15000
    config.num_mark = 13
    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt"
  else:
    config.batch_size = 1
    onfig.num_mark = 13
    config.result_dir = "result"
    config.ckpt_dir = "ckpt"
  return config
