#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


f1 = np.array([1,5,1,5,8])
f2 = np.array([2,5,4,3,1])
f3 = np.array([3,6,2,2,2])
f4 = np.array([4,7,3,1,2])


# In[4]:


def stand(x):
    mu = np.mean(x,axis =0)
    
    sigma = np.std(x,axis =0)
    
    x_norm = (x-mu)/sigma
    
    return x_norm


# In[5]:


f1_norm = stand(f1)
f2_norm = stand(f2)
f3_norm = stand(f3)
f4_norm = stand(f4)


# In[6]:


data = np.array([f1_norm,f2_norm,f3_norm,f4_norm])
cov = np.cov(data,bias = True)


# In[7]:


eigvalue,eigvector= np.linalg.eig(cov)
print(eigvalue,eigvector)


# In[8]:


def selecteig(eigvalue,eigvector,n_components):
    sorted_index = np.argsort(eigvalue)[::-1]
    sorted_eigenvalue = eigvalue[sorted_index]
    sorted_eigenvectors = eigvector[:,sorted_index]
    
    return sorted_eigenvectors[:,0:n_components]
            
            


# In[9]:


selected = selecteig(eigvalue,eigvector,2)
print(selected)


# In[10]:


PCA_result = np.dot(data.transpose(),selected)
print(PCA_result)


# In[ ]:





# In[ ]:




