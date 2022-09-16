#!/usr/bin/env python
# coding: utf-8

# ## TASK2 During The Intership at The Spark Foundation as Data Science and Business Analytics Intern.
#      Predict the optimum number of clusters and represent it visually for given "iris" dataset.
#     
# ***Author*** - Dinesh Kumar Kumawat

# In[1]:


#  Importing the all libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing  import MinMaxScaler


# In[2]:


# Read the dataset
iris = pd.read_csv("Iris.csv")
iris


# In[3]:


iris.drop("Id", axis=1, inplace = True)    # Id column won't help us in clustering.


# In[4]:


iris.head()  # To get 5 starting rows in  a dataset


# In[5]:


iris.tail()   # to get last 5 rows in a dataset.


# In[6]:


iris.isnull().sum()        # To count number of null datapoints.


# ***In above dataset, there is no null datapoints.***

# In[7]:


# To see the unique datapoints
iris.nunique()


# In[8]:


# To diplay the features of dataset.
iris.info()      


# In[9]:


# Stastical data analysis
iris.describe()


# In[10]:




iris1 = iris[iris.Species=='Iris-setosa']
iris2 = iris[iris.Species=='Iris-versicolor']
iris3 = iris[iris.Species=='Iris-virginica']

# plt.scatter(iris1["SepalLengthCm"], iris1["SepalWidthCm"], color="black", label="Iris-setosa")
# plt.scatter(iris2["SepalLengthCm"], iris2["SepalWidthCm"], color="Hotpink", label="Iris-versicolor")
# plt.scatter(iris3["SepalLengthCm"], iris3["SepalWidthCm"], color="Purple", label="Iris-verginica")

plt.scatter(iris1["PetalLengthCm"], iris1["PetalWidthCm"], color="r", label="Iris-setosa")
plt.scatter(iris2["PetalLengthCm"], iris2["PetalWidthCm"], color="b", label="Iris-versicolor")
plt.scatter(iris3["PetalLengthCm"], iris3["PetalWidthCm"], color="g", label="Iris-verginica")

plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.legend()
plt.show()


# In[11]:


iris.corr()


# ## Data Visualizations Plots

# In[12]:


sns.set(style="darkgrid")


# In[13]:


sns.histplot(iris, kde=True)


# In[19]:


sns.relplot(x='SepalLengthCm', y='SepalWidthCm', data=iris, size='SepalWidthCm', hue='SepalWidthCm', sizes=(50, 200),
            style="SepalLengthCm")


# In[15]:


sns.relplot(x='PetalLengthCm', y='PetalWidthCm', data=iris, size='PetalWidthCm', kind="line")


# In[16]:


sns.


# In[ ]:


sns.heatmap(iris.corr(), annot=True)


# ### Predicting the Optimum number of clusters

# In[ ]:


# Finding  the optimum number of clusters for k--means classification
x = iris.iloc[:, 0:4].values
sse = []      # Sum of squared error

for i in range(1, 11):
    kmeans =KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0) 
    kmeans.fit(x)
    sse.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'

plt.plot(range(1, 11), sse, marker="o")
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("sse")
plt.show()


# In[ ]:


#To see the Sum of Squared Error
sse


# Optimum clusters is where the elbow occurs.
# So we will choose the optimum number of cluser as 3.

# In[ ]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
y_kmeans


# In[ ]:


# To add the cluster column into the iris dataset.
iris["cluster"]=y_kmeans
iris


# In[ ]:


# To see the centroid of clusters
print(kmeans.cluster_centers_)


# In[ ]:


# Visualizing the cluster on the 3rd and 4th columns
plt.scatter(x[y_kmeans==0, 2], x[y_kmeans==0, 3], c="red", label="Iris-setosa")
plt.scatter(x[y_kmeans==1, 2], x[y_kmeans==1, 3], c="Hotpink", label="Iris-versicolour")
plt.scatter(x[y_kmeans==2, 2], x[y_kmeans==2, 3], c="green", label="Iris-virginica")
plt.legend()
# plt.show()

# plotting the centroid of clusters
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], 
            s = 100, c = 'yellow', label = 'Centroids')


# The Hotpink cluster looks okay there is a problem with red and green clusters we know they are not grouped correctly
# so this problem happened our scaling is not right.Our x-axis is scaled from let's say 0 to 7 and the range of y-axis
# is pretty narrow so it's like hardly 0.5 versus here is 1. So when we don't scale our features properly we might get 
# into this problem that's why we need to do some pre-processing and use MinMaxScaler to scale these two features and 
# then only we can run our algorithm.

# In[ ]:


scaler = MinMaxScaler()
scaler.fit(iris[['PetalWidthCm']])
iris['PetalWidthCm']=scaler.transform(iris[['PetalWidthCm']])
scaler.fit(iris[['PetalLengthCm']])
iris['PetalLengthCm']=scaler.transform(iris[['PetalLengthCm']])
scaler.fit(iris[['SepalWidthCm']])
iris['SepalWidthCm']=scaler.transform(iris[['SepalWidthCm']])
scaler.fit(iris[['SepalLengthCm']])
iris['SepalLengthCm']=scaler.transform(iris[['SepalLengthCm']])
iris


# ### Predicting the Optimum number of clusters

# In[ ]:


# Finding  the optimum number of clusters for k--means classification
x = iris.iloc[:, 0:4].values
sse = []      # Sum of squared error

for i in range(1, 11):
    kmeans =KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0) 
    kmeans.fit(x)
    sse.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'

plt.plot(range(1, 11), sse, marker="o")
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("sse")
plt.show()


# In[ ]:


#To see the Sum of Squared Error
sse


# Optimum clusters is where the elbow occurs. So we will choose the optimum number of cluser as 3.

# In[ ]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
y_kmeans


# In[ ]:


# To add the cluster column into the iris dataset.
iris["cluster"]=y_kmeans
iris


# In[ ]:


# To see the centroid of clusters
print(kmeans.cluster_centers_)


# In[ ]:


# Visualizing the cluster on the 3rd and 4th columns
plt.scatter(x[y_kmeans==0, 2], x[y_kmeans==0, 3], c="red", label="Iris-setosa")
plt.scatter(x[y_kmeans==1, 2], x[y_kmeans==1, 3], c="Hotpink", label="Iris-versicolour")
plt.scatter(x[y_kmeans==2, 2], x[y_kmeans==2, 3], c="green", label="Iris-virginica")
plt.legend()
# plt.show()

# plotting the centroid of clusters
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], 
            c = 'yellow', label = 'Centroids')

