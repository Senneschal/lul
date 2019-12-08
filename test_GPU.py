#!/usr/bin/env python
# coding: utf-8

# In[1]:


from distutils.version import LooseVersion
import tensorflow as tf


# In[2]:


tf.test.gpu_device_name()


# In[3]:


tf.test.is_gpu_available()


# In[ ]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[8]:


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

