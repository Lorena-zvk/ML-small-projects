#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px


# In[2]:


#load and prepare data


# In[3]:


test_data = pd.read_csv('ALS_TestingData_78.csv')
train_data = pd.read_csv('ALS_TrainingData_2223.csv')


# In[4]:


train_data.info()


# In[5]:


train_data.shape


# In[6]:


train_data.columns


# In[7]:


print(train_data)


# In[8]:


train_data.head()


# In[9]:


test_data.head()


# In[10]:


train_data.groupby('Age_mean').size()


# In[45]:


correlation = train_data.corr()
# display(correlation)
plt.figure(figsize=(103, 101))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
heatmap.figure.savefig("heatmap.png")


# In[11]:


#function to prepare the tsne

def prepare_tsne(nr, data):
    
    matrix  = TSNE(n_components = nr).fit_transform(data)
    data_matrix = pd.DataFrame(matrix)
    return(data_matrix)


# In[12]:


#visualize not yet normalized data for comparison
tsne_3d = prepare_tsne(3, train_data)

x = tsne_3d[[0]]
y = tsne_3d[[1]]
z = tsne_3d[[2]]
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 

ax.scatter3D(x, y, z, color = "blue")
plt.title("Data visualization in 3d")

plt.show()


# In[13]:


tsne_2d = prepare_tsne(2, train_data)

x = tsne_2d[[0]]
y = tsne_2d[[1]]

fig = plt.figure(figsize = (10, 7))
plt.scatter(x, y, c='red')
plt.show()


# In[14]:


#normalize the data


# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
train_data_normalized = scaler.transform(train_data)
train_data_normalized[1, :]


# In[16]:


print(train_data_normalized)


# In[17]:


train_data_normalized = pd.DataFrame(train_data_normalized, columns=train_data.columns)
train_data_normalized.head()


# In[18]:


#visualizing the data
#using t-SNE to visualize high dimentional data in a 3d and 2d space


# In[19]:


tsne_3d = prepare_tsne(3, train_data_normalized)

x = tsne_3d[[0]]
y = tsne_3d[[1]]
z = tsne_3d[[2]]
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 

ax.scatter3D(x, y, z, color = "green")
plt.title("Data visualization in 3d")

plt.show()


# In[20]:


tsne_2d = prepare_tsne(2, train_data_normalized)

x = tsne_2d[[0]]
y = tsne_2d[[1]]

fig = plt.figure(figsize = (10, 7))
plt.scatter(x, y, c='purple')
plt.show()


# In[21]:


#k-means algorithm


# In[33]:


data = train_data_normalized
kmeans = KMeans(n_clusters=7, max_iter=300,n_init=10,random_state=0)
kmeans.fit(data)


# In[34]:


kmeans.cluster_centers_


# In[35]:


#visualizing the clustered data in 2d and 3d space


# In[36]:


print(kmeans.labels_)


# In[37]:


tsne = prepare_tsne(2, train_data_normalized)
tsne['labels'] = kmeans.labels_


x = tsne_2d[[0]]
y = tsne_2d[[1]]

fig = plt.figure(figsize = (10, 7))
plt.scatter(x, y, c=tsne['labels'])
plt.show()


# In[38]:


tsne_3d = prepare_tsne(3, train_data_normalized)

tsne_3d['labels'] = kmeans.labels_

x = tsne_3d[[0]]
y = tsne_3d[[1]]
z = tsne_3d[[2]]
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 

ax.scatter3D(x, y, z, c = tsne_3d['labels'])
plt.title("Data visualization in 3d")

plt.show()


# In[28]:


#example of running k-means on not normalized data 


# In[30]:


data = train_data
kmeans = KMeans(n_clusters=4, max_iter=300,n_init=10,random_state=0)
kmeans.fit(data)


# In[31]:


tsne = prepare_tsne(2, train_data)
tsne['labels'] = kmeans.labels_


x = tsne_2d[[0]]
y = tsne_2d[[1]]

fig = plt.figure(figsize = (10, 7))
plt.scatter(x, y, c=tsne['labels'])
plt.show()


# In[32]:


tsne_3d = prepare_tsne(3, train_data)

tsne_3d['labels'] = kmeans.labels_

x = tsne_3d[[0]]
y = tsne_3d[[1]]
z = tsne_3d[[2]]
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 

ax.scatter3D(x, y, z, c = tsne_3d['labels'])
plt.title("Data visualization in 3d")

plt.show()


# In[ ]:


#finding out the optimal number of clusters with the elbow curve


# In[29]:


sse = []
normalized = preprocessing.normalize(train_data)
for i in range(1,15):
    
    Kmeans = KMeans(n_clusters=i,max_iter=300,n_init=10,random_state=0)
    Kmeans.fit(normalized)
    sse.append(Kmeans.inertia_)
    print("Cluster", i, "Inertia", Kmeans.inertia_)
    
plt.plot(range(1,15),sse)
plt.title('Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('sse') 
plt.show()


# In[ ]:


#based on the elbow curve we can see that the best nr. of clusters is 3


# In[52]:


#visualize the results for 3 clusters
data = train_data_normalized
kmeans = KMeans(n_clusters=3, max_iter=300,n_init=10,random_state=0)
kmeans.fit(data)


# In[53]:


tsne = prepare_tsne(2, data)
tsne['labels'] = kmeans.labels_


x = tsne_2d[[0]]
y = tsne_2d[[1]]

fig = plt.figure(figsize = (10, 7))
plt.scatter(x, y, c=tsne['labels'])
plt.show()


# In[54]:


tsne_3d = prepare_tsne(3, train_data_normalized)

tsne_3d['labels'] = kmeans.labels_

x = tsne_3d[[0]]
y = tsne_3d[[1]]
z = tsne_3d[[2]]
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 

ax.scatter3D(x, y, z, c = tsne_3d['labels'])
plt.title("Data visualization in 3d")

plt.show()


# In[48]:


#comparing kmeans based on variable changes
normalized = preprocessing.normalize(train_data)
kmeans = KMeans(n_clusters=4, max_iter=300,n_init=10,random_state=0)
kmeans.fit(normalized)
print(kmeans.inertia_)


# In[49]:


normalized = preprocessing.normalize(train_data)
kmeans = KMeans(n_clusters=4, max_iter=300,n_init=10,random_state=79)
kmeans.fit(normalized)
print(kmeans.inertia_)


# In[50]:


normalized = preprocessing.normalize(train_data)
kmeans = KMeans(n_clusters=4, max_iter=100,n_init=10,random_state=0)
kmeans.fit(normalized)
print(kmeans.inertia_)


# In[51]:


normalized = preprocessing.normalize(train_data)
kmeans = KMeans(n_clusters=4, max_iter=300,n_init=6,random_state=79)
kmeans.fit(normalized)
print(kmeans.inertia_)


# In[ ]:




