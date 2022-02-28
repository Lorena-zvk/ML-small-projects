#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('truckdata.txt')
X=data.iloc[:,0]
Y=data.iloc[:,1]


# In[3]:


plt.scatter(X, Y)
plt.show()


# In[26]:


data.head()


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=47)


# In[6]:


X_train.head()


# In[7]:


X_test.head()


# In[8]:


#gradient descent method


# In[9]:


c = 0
m = 0
n= float(len(X))
print(n)


# In[10]:


L=0.0001


# In[11]:


epochs=2000


# In[12]:


for i in range(epochs): 
    Y_pred = m*X_train +c   
    D_m = (-2/n)*sum(X_train * (Y_train - Y_pred))    
    D_c = (-2/n)*sum(Y_train - Y_pred)   
    m = m - L*D_m   
    c = c - L*D_c
print (m,c)


# In[13]:


plt.scatter(X_train, Y_train)
plt.plot([min(X_train), max(X_train)], [min(Y_pred), max(Y_pred)], color ='green') 
plt.show()


# In[14]:


plt.scatter(X_test, Y_test)
plt.plot([min(X_test), max(X_test)], [min(Y_pred), max(Y_pred)], color ='green') 
plt.show()


# In[30]:


residual=Y_train -Y_pred
residual_mean = statistics.mean(residual) 
Square_Residual_Sum=(residual*residual).sum() 
print (Square_Residual_Sum)


# In[31]:


plt.scatter(X_train, residual)  
plt.plot([min(X_train), max(X_train)], [residual_mean, residual_mean], color='red')
plt.show()


# In[17]:


#Least Squares Method


# In[18]:


n = np.size(data)
print(n)


# In[19]:


c = np.mean(Y)-np.mean(X)
print(c)


# In[20]:


m = n*np.sum(X*Y)-np.sum(X)*np.sum(Y)
m = m/(n*np.sum(X*X)-np.sum(X)*np.sum(X))
print(m)


# In[21]:


pred = m*X_train+c


# In[22]:


plt.scatter(X_train, Y_train)
plt.plot([min(X_train), max(X_train)], [min(pred), max(pred)], color ='green') 
plt.show()


# In[23]:


residual= Y_train -pred
residual_mean = statistics.mean(residual) 
Square_Residual_Sum=(residual*residual).sum() 
print (Square_Residual_Sum)


# In[24]:


plt.scatter(X_train, residual)
plt.plot([min(X_train), max(X_train)], [residual_mean, residual_mean], color='red')
plt.show()


# In[ ]:




