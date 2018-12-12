#!/usr/bin/env python3

import os
import tensorflow as tf
from models import RegNet
from config import get_config
from data import DataHandler
import nibabel
import numpy as np

config = get_config(is_train=True)

def main():
  sess = tf.Session()

  reg = RegNet(sess, config, "RegNet", is_train=True)
  dh = DataHandler("Curious_data", is_train=True)

  # Start training
  for step in range(config.iteration):
    batch_x, batch_y, mri_affine, us_affine, mri_shape, us_shape, mri_mark, us_mark = dh.sample_pair(config.batch_size)
    loss, grid_x, grid_y, grid_z =\
        reg.fit(batch_x, batch_y, mri_mark, us_mark, get_initial_map(mri_affine, us_affine, mri_shape, us_shape))
    grid_x = np.rint(grid_x).astype(int)
    grid_y = np.rint(grid_y).astype(int)
    grid_z = np.rint(grid_z).astype(int)

    # Calculate the mean target registration error
    mTREs = 0.
    for j in range(len(batch_x)):
      for i in range(len(mri_mark)):
          x_coord = mri_mark[j, i, :].astype(int)
          item = j * config.im_size[0] * config.im_size[1] * config.im_size[2] +\
              x_coord[0] * config.im_size[1] * config.im_size[2] +\
              x_coord[1] * config.im_size[2] + x_coord[2]
          y_coord = np.array([grid_x[item],grid_y[item], grid_z[item]]) + 1
          y_coord = y_coord * (us_shape[j,:3,0].astype(float)/config.im_size) 
          y_coord = nibabel.affines.apply_affine(us_affine[j, :], y_coord)
          y_mark = us_mark[j, i, :] * (us_shape[j,:3,0].astype(float)/config.im_size) 
          y_mark = nibabel.affines.apply_affine(us_affine[j, :], y_mark)
          mTREs += ((y_coord - y_mark) ** 2 ).sum() ** 0.5

    mTREs = mTREs / (len(mri_mark)*len(batch_x))
    print("iter {:>6d} : {} loss: {:>6f} mTREs: {:>6f}".format(step+1, loss, mTREs))

    # Save checkpoint
    if (step+1) % 100 == 0:
      reg.save(config.ckpt_dir, step+1)

# Initialize the deformation field
def get_initial_map(mri_affine, us_affine, mri_shape, us_shape):

    def _map_repeat(input_map, mod, im_size):
        if mod == 0:
            return input_map.repeat(im_size[1] * im_size[2])
        elif mod == 1:
            return np.tile(input_map.repeat(im_size[1]), im_size[2])
        else:
            return np.tile(input_map, im_size[1] * im_size[2])

    x_len, y_len, z_len = config.im_size
    mri_affine = mri_affine.astype(np.float64)
    us_affine = us_affine.astype(np.float64)
    mri_shape = mri_shape.astype(np.float64)
    us_shape = us_shape.astype(np.float64)
    map_value = np.tile(np.append(np.tile(np.arange(config.im_size[0], dtype=np.float64), 3),\
      np.full((config.im_size[0],),config.im_size[0],np.float64),axis=0),\
      config.batch_size).reshape((config.batch_size, 4, config.im_size[0]))

    map_value = np.matmul(mri_affine, map_value * mri_shape / config.im_size[0])
    map_value = np.matmul(np.linalg.inv(us_affine), map_value) / us_shape * config.im_size[0]

    base_map=np.empty([config.batch_size, 3 ,x_len*y_len*z_len], dtype=np.float64)
    for i in range(config.batch_size):
        for j in range(3):
            base_map[i,j,:] = _map_repeat(map_value[i,j,:],j,config.im_size) / (config.im_size[j]/2.) - 1

    base_map = base_map.reshape([config.batch_size, x_len, y_len, z_len, 3])
    return base_map.astype(np.float32)

if __name__ == "__main__":
  main()
