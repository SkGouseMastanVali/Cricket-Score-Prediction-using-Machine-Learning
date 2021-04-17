# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:34:42 2020

@author: gmvsh
"""

def custom_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)

#import matplotlib.pyplot as plt
#import seaborn as sns

# Importing the dataset
import pandas as pd
dataset = pd.read_csv('data/odi.csv')

#we are perofrming correlation between atrribute.which generally helps to eleminate not suitable
#attributes and unwanted attributes.
#here we are using seaborn for plotting graph (systamatic plan) in correlation and here seaborn
#is importing as sns
#using heatmap to see the correlation between feature and target variable and imported as plt.
'''
corelation=dataset[['runs','wickets','overs','runs_last_5','wickets_last_5','striker','non-striker','total']]
fig=plt.subplots(figsize=(14,16))
sns.heatmap(corelation.corr(),square=True,cbar=True,annot=True,cmap="GnBu",annot_kws={'size':8})
plt.title('Correlation between attributes')
plt.show() 
'''
import numpy as np

corr_matrix=dataset.corr().abs()
up=np.triu(np.ones_like(corr_matrix,dtype=bool))
data=corr_matrix.mask(up)

drop_col=[column for column in data.columns if any(data[column]>0.6)]
print(drop_col)
dropeddata=dataset.drop(dataset[drop_col],axis=1)
dropeddata.head()

'''
fig=plt.subplots(figsize=(14,16))
sns.heatmap(dropeddata.corr(),square=True,cbar=True,annot=True,cmap="GnBu",annot_kws={'size':8})
plt.title('Correlation between attributes')
plt.show() 
'''


#dropeddata.to_csv('dropeddata.csv',index=False)
'''
#here we can estimate how correlation between runs and total will effect
estimate_correlation=pd.pivot_table(dataset,index='total',values='runs')
estimate_correlation.plot(kind='bar',color='red',figsize=(10,10))
plt.xlabel("total")
plt.ylabel("runs")
plt.title("Total vs Runs")
plt.xticks(rotation=90)
plt.show()

#here we can estimate how correlation between wickets and total will effect
estimate_correlation=pd.pivot_table(dataset,index='total',values='wickets')
estimate_correlation.plot(kind='bar',color='red',figsize=(10,10))
plt.xlabel("total")
plt.ylabel("wickets")
plt.title("Total vs wickets")
plt.xticks(rotation=90)
plt.show()

#here we can estimate how correlation between overs and total will effect
estimate_correlation=pd.pivot_table(dataset,index='total',values='overs')
estimate_correlation.plot(kind='bar',color='red',figsize=(10,11))
plt.xlabel("total")
plt.ylabel("overs")
plt.title("Total vs Overs")
plt.xticks(rotation=90)
plt.show()

#here we can estimate how correlation between runs_last_5 and total will effect
estimate_correlation=pd.pivot_table(dataset,index='total',values='runs_last_5')
estimate_correlation.plot(kind='bar',color='red',figsize=(10,11))
plt.xlabel("total")
plt.ylabel("runs_last_5")
plt.title("Total vs runs_last_5")
plt.xticks(rotation=90)
plt.show()

#here we can estimate how correlation between wickets_last_5 and total will effect
estimate_correlation=pd.pivot_table(dataset,index='total',values='wickets_last_5')
estimate_correlation.plot(kind='bar',color='red',figsize=(10,11))
plt.xlabel("total")
plt.ylabel("wickets_last_5")
plt.title("Total vs wickets_last_5")
plt.xticks(rotation=90)
plt.show()
'''
#index starts from 0 to n
X = dropeddata.iloc[:,[7,8,9,10,11]].values
y = dropeddata.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
# Training the dataset
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100,max_features=None)
reg.fit(X_train,y_train)

# Testing the dataset on trained model
y_pred = reg.predict(X_test)
score1 = reg.score(X_test,y_test)*100
print("R square value:" , score1)
print("Custom accuracy for Random Forest:" , custom_accuracy(y_test,y_pred,20))

# Testing with a custom input
import numpy as np
new_prediction = reg.predict(sc.transform(np.array([[100,0,13,50,50]])))
print("Prediction score:" , new_prediction)




from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train,y_train)

# Testing the dataset on trained model
y_pred = lin.predict(X_test)
score2= lin.score(X_test,y_test)*100
print("R square value:" , score2)
print("Custom accuracy for Linear Regression:" , custom_accuracy(y_test,y_pred,20))

'''
from sklearn.linear_model import Ridge
rdregressor=Ridge(alpha=1,normalize=True)
rdregressor.fit(X_train,y_train)
y_pred = rdregressor.predict(X_train)
score3=rdregressor.score(X_train,y_train)*100
print("R Square value:",score3)
print("Custome accuracy for Ridge:",custom_accuracy(y_test,y_pred,20))


'''
from sklearn.linear_model import Lasso
lsregressor=Lasso(alpha=1,normalize=True)
lsregressor.fit(X_train, y_train)
y_pred = lsregressor.predict(X_test)
score4=lsregressor.score(X_train,y_train)*100
print("R Square value:",score4)
print("Custome accuracy for Ridge:",custom_accuracy(y_test,y_pred,20))
'''

'''
from sklearn.svm import SVR
svregressor=SVR()
svregressor.fit(X_train, y_train)
y_pred = svregressor.predict(X_test)
score5=svregressor.score(X_train,y_train)*100
print("R Square value:",score5)
print("Custome accuracy for SVR:",custom_accuracy(y_test,y_pred,20))

models=['Random Forest','Linear Regression','Ridge','Losso']
acc_score=[0.59,0.31,0.26,0.26]
plt.bar(models,acc_score,color=['green','pink','cyan','skyblue'])
plt.y_label("Accurate scores")
plt.title("Which model is the best accurate for inbalenced data")
plt.show()
'''

