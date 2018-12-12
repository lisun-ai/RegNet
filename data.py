#!/usr/bin/env python3
"""
Data loading for network
"""

import numpy as np
import nibabel, glob, os
from scipy import ndimage
from config import get_config

config = get_config(is_train=True)

def itensity_normalize(volume):
  pixels = volume[volume > 0]
  mean = pixels.mean()
  std  = pixels.std()
  out = (volume - mean)/std
  return out

def resize_ND_volume(volume, out_shape, order = 3):
  shape0=volume.shape
  assert(len(shape0) == len(out_shape))
  scale = [(out_shape[i] + 0.)/shape0[i] for i in range(len(shape0))]
  out_volume = ndimage.interpolation.zoom(volume, scale, order = order)
  return out_volume

class DataHandler(object):
  """
    Members :
      is_train - Options for sampling
      path - Data path
      data - a list of np.array w/ shape [batch_size, 150, 150, 150, 1]
  """
  def __init__(self, path, is_train):
    self.is_train = is_train
    self.path = path
    self.mri_data, self.us_data, self.mri_affine, self.us_affine, self.mri_shape, self.us_shape, self.mri_mark, self.us_mark = self._get_data()


  def _get_data(self):
    mods = ['before']#, 'during', 'after']
    mri_cases = glob.glob('../dataset/train/Case*-MRI-beforeUS.tag')
    mri_data = np.empty([len(mri_cases), config.im_size[0], config.im_size[1], config.im_size[2], 1], dtype=np.float32)
    us_data = mri_data.copy()
    mri_affine = np.empty([len(mri_cases), 4, 4], dtype=np.float32)
    us_affine = mri_affine.copy()
    mri_shape = np.ones([len(mri_cases), 4, 1], dtype=np.float32)
    us_shape = mri_shape.copy()
    mri_mark = np.empty([len(mri_cases), config.num_mark, 3], dtype=np.float32)
    us_mark = mri_mark.copy()
    for i in range(len(mri_cases)):
      mri_case=mri_cases[i]
      case_id = os.path.basename(mri_case).split('-')[0]
      mod = os.path.basename(mri_case).split('-')[2][:-6]
      mri_img = nibabel.load('../dataset/train/'+case_id+'-FLAIR.nii.gz')
      mri_img_data=mri_img.get_data()
      mri_shape[i,:3, 0] = mri_img_data.shape
      mri_img_data=resize_ND_volume(mri_img_data, config.im_size)
      mri_img_data=itensity_normalize(mri_img_data)
      mri_affine[i,:,:] = mri_img.get_affine()
      mri_data[i,:,:,:,0] = mri_img_data.copy()
      us_img=nibabel.load('../dataset/train/'+case_id+'-US-'+mod+'.nii.gz')
      us_img_data = us_img.get_data()
      us_shape[i,:3, 0] = us_img_data.shape
      us_img_data=resize_ND_volume(us_img_data, config.im_size)
      us_img_data=itensity_normalize(us_img_data)
      us_data[i,:,:,:,0] = us_img_data
      us_affine[i,:,:] = us_img.get_affine()
      FILE=open(mri_case, 'r')
      for j in range(config.num_mark):
          landmark = [float(x) for x in FILE.readline().strip("\n").split("\t")]
          mri_mark[i,j,:] = landmark[:3]
          us_mark[i,j,:] = landmark[3:]
      FILE.close()
    return mri_data, us_data, mri_affine, us_affine, mri_shape, us_shape, mri_mark, us_mark

  def sample_pair(self, batch_size):
    
    choice = np.random.choice(len(self.mri_data), batch_size)
    x = self.mri_data[choice]
    y = self.us_data[choice]
    # Add random gussian noise to augment data
    x += np.random.normal(0., 0.2, x.shape)
    y += np.random.normal(0., 0.2, y.shape)
    x_aff = self.mri_affine[choice]
    y_aff = self.us_affine[choice]
    x_shape = self.mri_shape[choice]
    y_shape = self.us_shape[choice]
    x_mark = self.mri_mark[choice]
    y_mark = self.us_mark[choice]
    return x, y, x_aff, y_aff, x_shape, y_shape, x_mark, y_mark
