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

#null values are not exist in the dataset, if exist replace with mean value
'''print(dataset.isnull().sum()) #checking for missing values or null values 
#dataset.wickets=dataset.wickets.fillna(dataset.wickets.mean())'''

'''for col in dataset.columns:  #value counts
    print(col.value_count())'''


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
'''
'''
# Testing with a custom input
import numpy as np
new_prediction = reg.predict(sc.transform(np.array([[100,0,13,50,50]])))
print("Prediction score:" , new_prediction)

# leanear regression custome is 43
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train,y_train)
y_pred = lin.predict(X_test)    # Testing the dataset on trained model
score2= lin.score(X_test,y_test)*100
print("R square value:" , score2)
print("Custom accuracy for Linear Regression:" , custom_accuracy(y_test,y_pred,20))
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
'''
'''
'''
#r suare = 94.34 and cusome is 86.42
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1,weights='distance')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score8=classifier.score(X_train,y_train)*100
print("R Squre value:",score8)
print("Custome accuracy for KNeighborsClassifier:",custom_accuracy(y_test,y_pred,20))


'''

'''
import matplotlib.pyplot as plt
models=['RF','LR','DTC','K-NC']
acc_score=[0.77,0.43,0.79,0.86]
plt.bar(models,acc_score,color=['green','pink','cyan','skyblue'])
plt.title('Which model is the best accurate for inbalenced data')
plt.ylabel("Accurate scores")
plt.xlabel("Algorithms")
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.show()
print("RF---->Random Forest Regressor")
print("LR---->Linear Regressor")
print("DTC---->Decision Tree Classifier")
print("K-NC---->K-Neighbors Classifier")



