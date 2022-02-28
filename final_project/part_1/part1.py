#!/usr/bin/env python
# coding: utf-8

# In[1]:


#neural network
#part 1 of portfolio
import numpy as np
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt


# In[2]:


class NN:
    
    def __init__(self, layers, learning_rate):
        
      
        self.weights = []
        for i in range(0, len(layers)-2):
            w = np.random.randn(layers[i]+1, layers[i+1]+1)
            w = w / np.sqrt(layers[i])
            self.weights.append(w)
        w = np.random.randn(layers[-2]+1, layers[-1])
        w = w / np.sqrt(layers[-2])
        self.weights.append(w)
        
        self.learning_rate = learning_rate
        self.layers = layers
    
    def sigmoid(self, x):
    
        return 1/(1+np.exp(-x))
    
    def sigmoid_deriv(self, x):
    
        return x*(1-x)
    
    def forward_feed(self, x, y):
        
        a = [np.atleast_2d(x)]
        for layer in range(0, len(self.weights)):
            
            net = a[layer].dot(self.weights[layer])
            out = self.sigmoid(net)
            a.append(out)
        return a
    def backpropagation(self, a, x, y):
        
        error = a[-1] - y
        d = [error * self.sigmoid_deriv(a[-1])]
        for layer in range(len(a)-2, 0, -1):
            delta = d[-1].dot(self.weights[layer].T)
            delta = delta * self.sigmoid_deriv(a[layer])
            d.append(delta)
        d = d[::-1]
        
        #update the weight matrix
        for layer in range(0, len(self.weights)):
            self.weights[layer] += -self.learning_rate*a[layer].T.dot(d[layer])
              
    def predict(self, x, addBias = True):
        
        prediction = np.atleast_2d(x)
        #adding bias
        if addBias:
            prediction = np.c_[prediction, np.ones((prediction.shape[0]))]
    
        
        for layer in range(0, len(self.weights)):
            prediction = self.sigmoid(np.dot(prediction, self.weights[layer]))
        return prediction
    
    def calculate_loss(self, x, target):
        
        target = np.atleast_2d(target)
        predictions = self.predict(x, addBias = False)
        loss = 0.5 * np.sum((predictions - target)**2)
        return loss
    
    def fit(self, x, y):
        
        #adding the bias into the data
        #as a column of 1's at the end of the data matrix
        x = np.c_[x, np.ones((x.shape[0]))]
        loss_array = []
        epochs_array = []
        stop_condition = 1
        epoch = 0
        while stop_condition:
            for (X, Y) in zip(x, y):
                
                a = self.forward_feed(X, Y)
                self.backpropagation(a, X, Y)
            if epoch == 0 or (epoch + 1) % 1000 == 0:
                loss = self.calculate_loss(x, y)
                epochs_array.append(epoch)
                loss_array.append(loss)
                print("epoch={}, loss={:.7f}".format(epoch + 1, loss))
            pred = self.predict(x, addBias=False)
            stop_condition = 0
            for i in range(0, len(pred)):
                if abs(pred[i]-y[i])>=0.05:
                    stop_condition = 1
            epoch+=1
        print(epoch)    
        plt.plot(epochs_array, loss_array)
        plt.show()
            


# In[3]:


x = np.array([ 
    [0, 0, 0, 0], 
    [0, 0, 1, 0], 
    [0, 0, 1, 1], 
    [0, 1, 0, 0], 
    [0, 1, 0, 1], 
    [0, 1, 1, 0], 
    [0, 1, 1, 1], 
    [1, 0, 0, 0], 
    [1, 0, 0, 1],
    [0, 1, 1, 0]
])
y = np.array([[0], [1], [0], [1], [0], [0],[1], [1], [0], [0]])


# In[4]:


nn = NN([4, 4, 1], 0.05)
nn.fit(x, y)


# In[5]:


nn = NN([4, 4, 1], 0.10)
nn.fit(x, y)


# In[6]:


nn = NN([4, 4, 1], 0.15)
nn.fit(x, y)


# In[7]:


nn = NN([4, 4, 1], 0.2)
nn.fit(x, y)


# In[8]:


nn = NN([4, 4, 1], 0.25)
nn.fit(x, y)


# In[9]:


nn = NN([4, 4, 1], 0.3)
nn.fit(x, y)


# In[10]:


nn = NN([4, 4, 1], 0.35)
nn.fit(x, y)


# In[11]:


nn = NN([4, 4, 1], 0.4)
nn.fit(x, y)


# In[12]:


nn = NN([4, 4, 1], 0.45)
nn.fit(x, y)


# In[13]:


nn = NN([4, 4, 1], 0.5)
nn.fit(x, y)


# In[14]:


epochs = [28275, 16142, 30923, 5091, 16200, 5196, 9180, 3598, 3693, 3722]
lr = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
plt.plot(lr, epochs)
plt.show()


# In[15]:


nn = NN([4, 4, 1], 0.05)
nn.fit(x, y)


# In[16]:


x = np.array([
    [0, 0, 0, 0], 
    [0, 0, 0, 1], 
    [0, 0, 1, 0], 
    [0, 0, 1, 1], 
    [0, 1, 0, 0], 
    [0, 1, 0, 1], 
    [0, 1, 1, 0], 
    [0, 1, 1, 1], 
    [1, 0, 0, 0], 
    [1, 0, 0, 1], 
    [1, 0, 1, 0], 
    [1, 0, 1, 1], 
    [1, 1, 0, 0], 
    [1, 1, 0, 1], 
    [1, 1, 1, 1]
])
y = np.array([[0], [1], [1], [0], [1], [0], [0],[1], [1], [0], [0], [1], [0], [1], [0]])


# In[17]:


y_pred = []
for (x, target) in zip(x, y):
    pred = nn.predict(x)
    step = 1 if pred > 0.5 else 0
    y_pred.append(step)
    print(x, target[0], pred, step)


# In[18]:


from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:




