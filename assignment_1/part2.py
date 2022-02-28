#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from sklearn import datasets


# In[2]:


iris = pd.read_csv('iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
print(iris.head())


# In[3]:


colors = {'Iris-setosa':'r', 'Iris-versicolor':'g', 'Iris-virginica':'b'}
fig, ax = plt.subplots()
for i in range(len(iris['sepal_length'])):
    ax.scatter(iris['sepal_length'][i], iris['sepal_width'][i],color=colors[iris['class'][i]])


# In[4]:


X = iris.drop(columns=['class'])
X.head()


# In[5]:


iris.groupby('class').size()


# In[6]:


feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris[feature_columns].values
y = iris['class'].values


# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify=y)


# In[9]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7) 

knn.fit(X_train, y_train)


# In[10]:


y_pred = knn.predict(X_test)


# In[11]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




