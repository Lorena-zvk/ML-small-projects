#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


# In[2]:


digits = datasets.load_digits()


# In[3]:


dir(digits)


# In[4]:


digits.images.shape


# In[5]:


digits.target


# In[6]:


digits.data


# In[7]:


print(digits.images[0])


# In[8]:


print(digits.images[107])


# In[9]:


plt.imshow(digits.images[0], cmap='binary')
plt.show()


# In[10]:


plt.imshow(digits.images[177], cmap='binary')
plt.show()


# In[11]:


#split dataset
y = digits.target
x = digits.images.reshape((len(digits.images), -1))


# In[12]:


print(x[0])


# In[ ]:





# In[13]:


print(x[0])


# In[14]:


#split train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=47)
print(x_train)


# In[ ]:





# In[15]:


#neural network with 1 hidden layer


# In[16]:


from sklearn.neural_network import MLPClassifier
mlp1 = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic',
                    random_state=5, learning_rate_init=.05, verbose='True')


# In[17]:


mlp1.fit(x_train, y_train)


# In[19]:


predictions = mlp1.predict(x_test)
predictions[:50] 


# In[20]:


y_test[:50]


# In[23]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[24]:


mlp2 = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic',
                    random_state=5, learning_rate_init=.5)


# In[25]:


mlp2.fit(x_train, y_train)


# In[26]:


predictions = mlp2.predict(x_test)


# In[27]:


accuracy_score(y_test, predictions)


# In[28]:


#neural network with more than 1 hidden layer
mlp3 = MLPClassifier(hidden_layer_sizes=(15, 10), activation='logistic',
                    random_state=5, learning_rate_init=.05)

mlp3.fit(x_train, y_train)
predictions = mlp3.predict(x_test)
accuracy_score(y_test, predictions)


# In[33]:


#neural network with a large number of nodes in the hidden layer
mlp4 = MLPClassifier(hidden_layer_sizes=(150, 100), activation='logistic',
                    random_state=5, learning_rate_init=.05)

mlp4.fit(x_train, y_train)
predictions = mlp4.predict(x_test)
accuracy_score(y_test, predictions)


# In[39]:


mlp5 = MLPClassifier(hidden_layer_sizes=(19,), activation='logistic',
                    random_state=5, learning_rate_init=.05)

mlp5.fit(x_train, y_train)
predictions = mlp5.predict(x_test)
accuracy_score(y_test, predictions)


# In[31]:


#flipping the train and test data


# In[34]:


mlp6 = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic',
                    random_state=5, learning_rate_init=.05)

mlp6.fit(x_test, y_test)
predictions = mlp6.predict(x_train)
accuracy_score(y_train, predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




