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
    

#we are perofrming correlation between atrribute.we can perform correlation with numerical data only.which generally helps to eleminate not suitable
#attributes and unwanted attributes.
#here we are using seaborn for plotting graph (systamatic plan) in correlation and here seaborn
#is importing as sns
#using heatmap to see the correlation between feature and target variable and imported as plt.

#print(dataset.shape)
#print(dataset.isnull().sum())
#sns.heatmap(dataset.isnull())
print(dataset.head())
'''  
import seaborn as sns  
print(dataset.corr())   
sns.heatmap(dataset.corr())
        or
import matplotlib.pyplot as plt
corelation=dataset[['runs','wickets','overs','runs_last_5','wickets_last_5','striker','non-striker','total']]
fig=plt.subplots(figsize=(14,16))
sns.heatmap(corelation.corr(),square=True,cbar=True,annot=True,cmap="GnBu",annot_kws={'size':8})
plt.title('Correlation between attributes')
plt.show() 
'''

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
'''
import seaborn as sns

#sns.boxplot(dataset['mid'])   #cheching outliers are present in mid or not by plotting diagram
#sns.boxplot(dataset['date'])
#sns.boxplot(dataset['venue'])
#sns.boxplot(dataset['bat_team'])
#sns.boxplot(dataset['bowl_team'])
#sns.boxplot(dataset['batsman'])
#sns.boxplot(dataset['bowler'])
#sns.boxplot(dataset['runs'])       # MOST CONTAIN OUTLIERS
#sns.boxplot(dataset['wickets'])
#sns.boxplot(dataset['overs'])
#sns.boxplot(dataset['runs_last_5'])
#sns.boxplot(dataset['wickets_last_5'])
#sns.boxplot(dataset['striker'])
#sns.boxplot(dataset['non-striker'])
#sns.boxplot(dataset['total'])


'''
'''
REMOVING OUTLIERS 
'''   

'''
print(dataset['runs'].describe())
print(dataset['wickets'].describe())
print(dataset['overs'].describe())
print(dataset['runs_last_5'].describe())
print(dataset['wickets_last_5'].describe())
print(dataset['striker'].describe())
print(dataset['non-striker'].describe())
print(dataset['total'].describe())
'''

'''             ANOTHER WAY TO DELETE OUTLIERS

#print(dataset['total'].describe())
q1=dataset.striker.quantile(0.25)
q3=dataset.striker.quantile(0.75)
iqr=q3-q1
l=q1-1.5*iqr
u=q3+1.5*iqr
temp=dataset[(dataset.striker>l) & (dataset.striker<u)]
print(temp)
sns.boxplot(temp['striker']).set_title("without outliers")

ds_new=dataset[(dataset.runs<=168) & (dataset.wickets<=4) & (dataset.runs_last_5<=29) & (dataset.wickets_last_5<=1)  & (dataset.total<=298)]
print(ds_new.shape)
'''
#sns.boxplot(ds_new['runs']).set_title("without outliers")
#sns.boxplot(ds_new['wickets']).set_title("without outliers")
#sns.boxplot(ds_new['runs_last_5']).set_title("without outliers")
#sns.boxplot(ds_new['wickets_last_5']).set_title("without outliers")
#sns.boxplot(ds_new['striker']).set_title("without outliers")
#sns.boxplot(ds_new['non-striker']).set_title("without outliers")
#sns.boxplot(ds_new['total']).set_title("without outliers")



'''#BUILDING CHI_SQUARE
from sklearn.feature_selection import chi2
a=dataset[['mid','date','venue','bat_team','bowl_team']]
b=dataset['total']
chi_scores=chi2(a,b)
p_values = pd.series(chi_scores[1],index = a.columns)
p_values.sort_values(ascending=False,inplace=True)
p_values.plot()
'''
'''





'''
'''
#FINDING OUTLAYERS (unrelated data to my dataset...so remove them)
import seaborn as sns
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

sns.boxplot(x=dataset['wickets_last_5'])

'''
'''

# Rfr R square value 79 custome is 77
from sklearn.ensemble import RandomForestRegressor # Training the dataset
reg = RandomForestRegressor(n_estimators=100,max_features=None)
reg.fit(X_train,y_train)

# Testing the dataset on trained model
y_pred = reg.predict(X_test)
score = reg.score(X_test,y_test)*100
print("R square value:" , score)
print("Custom accuracy for Random forest:" , custom_accuracy(y_test,y_pred,20))

# Testing with a custom input
import numpy as np
new_prediction = reg.predict(sc.transform(np.array([[100,0,13,50,50]])))
print("Prediction score:" , new_prediction)
'''
'''
# leanear regression custome is 43
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train,y_train)
y_pred = lin.predict(X_test)    # Testing the dataset on trained model
score2= lin.score(X_test,y_test)*100
print("R square value:" , score2)
print("Custom accuracy for Linear Regression:" , custom_accuracy(y_test,y_pred,20))
'''
'''# losso r square is 0 and custom is 26.6
from sklearn.linear_model import Lasso
lsregressor=Lasso(alpha=1,normalize=True)
lsregressor.fit(X_train, y_train)
y_pred = lsregressor.predict(X_test)
score4=lsregressor.score(X_train,y_train)*100
print("R Square value:",score4)
print("Custome accuracy for Lasso:",custom_accuracy(y_test,y_pred,20))
'''
'''
#r square : 57 and custome is 49
from sklearn.svm import SVR
svregressor=SVR(kernel='rbf')
svregressor.fit(X_train, y_train)
y_pred = svregressor.predict(X_test)
score5=svregressor.score(X_train,y_train)*100
print("R Square value:",score5)
print("Custome accuracy for SVR:",custom_accuracy(y_test,y_pred,20))
# Testing with a custom input
import numpy as np
new_prediction = svregressor.predict(sc.transform(np.array([[100,0,13,50,50]])))
print("Prediction score:" , new_prediction)
'''
'''
#gnb R sq is 3.30 and custom is 36.59
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
score6=gnb.score(X_train,y_train)*100
print("R Square value:",score6)
print("Custome accuracy for Gaussian Naive Bayes:",custom_accuracy(y_test,y_pred,20))
# Testing with a custom input
import numpy as np
new_prediction = gnb.predict(sc.transform(np.array([[100,0,13,50,50]])))
print("Prediction score:" , new_prediction)
'''

'''
#dtc r square is 94.411 and custom is 79.77

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(max_depth=100)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)                    #finding R square value with 25% of test data
score7=dtc.score(X_train,y_train)*100
print("R Square value:",score7)
print("Custome accuracy for Decision Trees Classifier:",custom_accuracy(y_test,y_pred,20))
# Testing with a custom input
import numpy as np
new_prediction = dtc.predict(sc.transform(np.array([[200,2,30,100,100]])))
print("Prediction score:" , new_prediction)
'''

'''
# dtr r square 94.44 and custom 78.08
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(max_depth=75)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
score7=dtr.score(X_train,y_train)*100
print("R Square value:",score7)
print("Custome accuracy for Gaussian Decision Trees Classifier:",custom_accuracy(y_test,y_pred,20))
'''


#r suare = 94.34 and cusome is 86.42
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1,weights='distance')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score8=classifier.score(X_train,y_train)*100
print("R Squre value:",score8)
print("Custome accuracy for KNeighborsClassifier:",custom_accuracy(y_test,y_pred,20))
# Testing with a custom input
import numpy as np
new_prediction = classifier.predict(sc.transform(np.array([[200,2,30,100,100]])))
print("Prediction score:" , new_prediction)



'''
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
new_prediction = neigh.predict(sc.transform(np.array([[200,2,30,100,100]])))
print("Prediction score:" , new_prediction)

'''

models=['RandomForestRegression','LinearRegression','Losso','GaussianNB','DecisionTreeRegressor','KNeighborsRegression','SupportVectorMachine']
acc_score=[0.77,0.43,0.27,0.37,0.78,0.87,0.49]
plt.rcParams['figure.figsize']=(15,7)
plt.bar(models,acc_score,color=['green','pink','cyan','skyblue','orange','lime','blue'])
plt.ylabel("Accurate scores")
plt.title("Which model is the best accurate for inbalenced data")
plt.show()

