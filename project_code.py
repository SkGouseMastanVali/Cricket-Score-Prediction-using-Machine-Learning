# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 21:01:24 2020

@author: gmvsh
"""
import matplotlib.pyplot as plt
import seaborn as sns


def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)

# Importing the dataset
import pandas as pd
dataset = pd.read_csv('data/odi.csv')
    

#
X = dataset.iloc[:,[7,8,9,12,13]].values     #input features
y = dataset.iloc[:, 14].values              #output featues           

# Splitting the dataset into the Training set(75%) and Test set(25%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler        #standard scale prepocessing is doing here and next is to train the data with different algos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
    # Let's say, components = 2 \n
pca = PCA()
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)


#null values are not exist in the dataset, if exist replace with mean value
'''print(dataset.isnull().sum()) #checking for missing values or null values 
#dataset.wickets=dataset.wickets.fillna(dataset.wickets.mean())'''

'''for col in dataset.columns:  #value counts
    print(col.value_count())'''

'''            FINDNING OUTLIERS        '''
#print(dataset)
#print((dataset).shape) 

#FINDING OUTLAYERS (unrelated data to my dataset...so remove them)

sns.boxplot(x=dataset['runs'])
sns.boxplot(x=dataset['wickets'])
sns.boxplot(x=dataset['overs'])
sns.boxplot(x=dataset['runs_last_5'])
sns.boxplot(x=dataset['wickets_last_5'])
sns.boxplot(x=dataset['striker'])
sns.boxplot(x=dataset['non-striker'])
sns.boxplot(x=dataset['total'])

#DELETING OUTLIERS FROM DATASET
def remove_outliers(dataset,col_name):
    q1=dataset[col_name].quantile(0.25)
    q3=dataset[col_name].quantile(0.75)
    iqr=q3-q1
    fence_low=q1-1.5*iqr
    fence_high=q3+1.5*iqr
    dataset_out=dataset.loc[(dataset[col_name]>fence_low)&(dataset[col_name]<fence_high)]
    return dataset_out
l=['runs','wickets','overs','runs_last_5','wickets_last_5','striker','non-striker','total']
for i in l:
    outliers_removed = remove_outliers(dataset,i)
print(outliers_removed)

#dropeddata.to_csv('dropeddata.csv',index=False)
outliers_removed.to_csv('outliers_removed.csv',index=False)


#r square= 88.95 and custome is 86.42
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=1)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
score9= neigh.score(X_train,y_train)*100
print("R Squre value:",score9)
print("Custome accuracy for KNeighborsRegressor:",custom_accuracy(y_test,y_pred,20))
# Testing with a custom input
import numpy as np
new_prediction = neigh.predict(sc.transform(np.array([[100,0,13,50,50]])))
print("Prediction score:" , new_prediction)



models=['RandomForestRegression','LinearRegression','Losso','GaussianNB','DecisionTreeRegressor','KNeighborsRegression','SupportVectorMachine']
acc_score=[0.77,0.43,0.27,0.37,0.78,0.87,0.49]
plt.rcParams['figure.figsize']=(15,7)
plt.bar(models,acc_score,color=['green','pink','cyan','skyblue','orange','lime','blue'])
plt.ylabel("Accurate scores")
plt.title("Which model is the best accurate for inbalenced data")
plt.show()

