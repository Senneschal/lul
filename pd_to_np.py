#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import os


# In[3]:


data=pd.read_csv("20190805_322_w1_s2_ground_truth.csv")
data.head()


# In[4]:


data = data[data['Segment']==5]


# In[5]:


data.pop('Segment')
print('shape: '+str(data.shape[0]))


# In[6]:


mask = np.zeros((286,312,512,35), dtype=np.uint8)


# In[7]:


values = data.values
N = data.values.shape[0]
print(N)

for i in range(N):
    point = values[i].astype(int)
    mask[point[0],point[1],point[2],point[3]] = 1

print(mask.sum())


# In[8]:


mask.shape


# In[9]:


b = []


# In[11]:


for i in range(286):
    if mask[i].sum()==0:
        b.append(i)


# In[12]:


b


# In[13]:


len(b)


# In[14]:


fixed_mask = np.zeros((286-35,312,512,35), dtype=np.uint8)


# In[15]:


j=0
for i in range(286):
    if mask[i].sum()>0:
        fixed_mask[j]=mask[i]
        j+=1


# In[16]:


fixed_mask.shape


# In[19]:


np.savez_compressed('mask1', fixed_mask)

