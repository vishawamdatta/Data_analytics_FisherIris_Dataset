# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:19:16 2020

@author: VD
"""
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
irisdata = load_iris()
import pandas as pd 
import seaborn as sns
import numpy as np
##house_price.data = preprocessing.scale(house_price.data)
df = pd.DataFrame(irisdata.data,
columns=irisdata.feature_names)
df['type']=irisdata.target


x=irisdata.data
y=irisdata.target


##histogram
sns.histplot(data=df,x='sepal length (cm)',y='sepal width (cm)',hue='type',palette="deep",bins=10)
plt.show()



##plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c ="blue") 
##plt.show()




## scatter plots 
##df.plot.scatter(x='petal length (cm)',y='petal width (cm)')

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df,x='petal length (cm)', y='petal width (cm)', hue='type',palette="deep") 
plt.show()

##ax.scatter(df['petal length (cm)'], df['petal width (cm)'], c = pd.Categorical(df['type']).codes, cmap='tab20b')
##plt.show()





## box plot
boxplot = df.boxplot(column=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)'])

fig1, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='type',y='petal length (cm)',data=df)
plt.show()

fig2, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='type',y='petal width (cm)',data=df)
plt.show()

fig3, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='type',y='sepal length (cm)',data=df)
plt.show()

fig4, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='type',y='sepal width (cm)',data=df)
plt.show()


## naive bais
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)
znaeve=clf.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
confusionnaeve=confusion_matrix(y_test, znaeve)
target_names=['setosa','versicolor','virginica']
classnaeve=classification_report(y_test, znaeve, target_names=target_names)

## logistic regression
from sklearn.linear_model import LogisticRegression 
clf1 = LogisticRegression(random_state=0,multi_class='ovr')
clf1.fit(x_train, y_train)
zlogistic=clf1.predict(x_test)
confusionlogistic=confusion_matrix(y_test, zlogistic)
classlogistic=classification_report(y_test, zlogistic, target_names=target_names)


## k means
from sklearn.cluster import KMeans
clf2 = KMeans(n_clusters=3, random_state=0)
zkmeans=clf2.fit_predict(x)
for i in range(150):
    if zkmeans[i]==1:
        zkmeans[i]=0
    elif zkmeans[i]==2:
        zkmeans[i]=1
    else:
        zkmeans[i]=2
        
confusionkmeans=confusion_matrix(y, zkmeans)
classkmeans=classification_report(y,zkmeans,target_names=target_names)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_r = pca.fit(x).transform(x)
y_final=y.reshape(150,1)

pcanp=np.hstack((X_r,y_final))
index_values=[i for i in range(150)]
cols=['axis 1','axis 2','type']
dffinal = pd.DataFrame(data = pcanp,  index = index_values,   columns = cols) 
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=dffinal,x='axis 1', y='axis 2', hue='type',palette="deep")
plt.show()


import plotly.express as px 

pca = PCA(n_components=3)
X_r = pca.fit(x).transform(x)
y_final=y.reshape(150,1)

pcanp=np.hstack((X_r,y_final))
index_values=[i for i in range(150)]
cols=['axis 1','axis 2','axis 3','type']
dffinal = pd.DataFrame(data = pcanp,  index = index_values,   columns = cols)

fig = px.scatter_3d(dffinal, x='axis 1', y='axis 2', z='axis 3',
              color='type',color_discrete_sequence=['blue','orange','green'])
fig.show()

