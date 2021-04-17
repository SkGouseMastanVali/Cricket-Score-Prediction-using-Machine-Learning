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

X = dataset.iloc[:,[7,8,9,12,13]].values
y = dataset.iloc[:, 14].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
    # Let's say, components = 2 \n
pca = PCA()
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)


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

        # DELETING PROCESS OF OUTLIERS FROM DATASET
def remove_outliers(dataset,col_name):
    Q1=dataset[col_name].quantile(0.25)     # 25% of data is below certain value in dataset     (To know them dataset.describe())
    Q3=dataset[col_name].quantile(0.75)     # 75%  is below some values in dataset....
    IQR=Q3-Q1                               # findinding IQR value
    lower_limit=Q1-1.5*IQR                  # finding upper and lower limits
    upper_limit=Q3+1.5*IQR
    
    outliers=dataset.loc[(dataset[col_name]>lower_limit)&(dataset[col_name]<upper_limit)]  # removing outliers and storing in new dataframe called outliers
    return outliers
l=['runs','wickets','overs','runs_last_5','wickets_last_5','striker','non-striker','total']
for i in l:
    outliers_removed = remove_outliers(dataset,i)
print(outliers_removed)

#dropeddata.to_csv('dropeddata.csv',index=False)
outliers_removed.to_csv('outliers_removed.csv',index=False)
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

# losso r square is 0 and custom is 26.6
from sklearn.linear_model import Lasso
lsregressor=Lasso(alpha=1,normalize=True)
lsregressor.fit(X_train, y_train)
y_pred = lsregressor.predict(X_test)
score4=lsregressor.score(X_train,y_train)*100
print("R Square value:",score4)
print("Custome accuracy for Lasso:",custom_accuracy(y_test,y_pred,20))

from sklearn.svm import SVR
svregressor=SVR()
svregressor.fit(X_train, y_train)
y_pred = svregressor.predict(X_test)
score5=svregressor.score(X_train,y_train)*100
print("R Square value:",score5)
print("Custome accuracy for SVR:",custom_accuracy(y_test,y_pred,20))
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

'''
#dtc r square is 94.411 and custom is 79.77

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(max_depth=100)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
score7=dtc.score(X_train,y_train)*100
print("R Square value:",score7)
print("Custome accuracy for Gaussian Decision Trees Classifier:",custom_accuracy(y_test,y_pred,20))
# Testing with a custom input
import numpy as np
new_prediction = dtc.predict(sc.transform(np.array([[100,0,13,50,50]])))
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
import matplotlib.pyplot as plt
models=['Random Forest','Linear Regression','Ridge','Losso','SVR','GaussianNB']
acc_score=[0.77,0.43,0.26,0.24,0.40,0.36]
plt.bar(models,acc_score,color=['green','pink','cyan','skyblue','orange','lime'])
plt.y_label("Accurate scores")
plt.title("Which model is the best accurate for inbalenced data")
plt.show()

