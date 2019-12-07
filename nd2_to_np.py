#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
import copy
from pims import ND2_Reader


# In[3]:


def prep_frames(frames, c=1):
    frames.iter_axes = 't'  # 't' is the default already
    frames.bundle_axes = 'yxz'  # when 'z' is available, this will be default
    frames.default_coords['c'] = c  # 0 is the default setting
    return frames

def export_npz(frames, images_path, c=1):
    framesList = []
    for frame in frames:
        frameArray = np.array(frame)
        framesList.append(frameArray)
    images_path = images_path.replace('.nd2', "_color"+str(c)+".npz")
    np.savez(images_path, framesList)
    return np.array(framesList)


# In[5]:


for file in os.listdir("./videos"):
    if file.endswith(".nd2"):
        images_path = os.path.join("data", file)
        frames = ND2_Reader(images_path)
        frames = prep_frames(frames, 1)
        framesArray = export_npz(frames, images_path, 1)
        print("[Success] {} succesfully saved to npz ! ".format(images_path))


# In[ ]:




