import time
start_time = time.time()
start_time1 = time.clock()
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier


n_estimators = 400
# A learning rate of 1. may not be optimal for both SAMME and SAMME.R
learning_rate = 1.

datasets = pd.read_csv('Full_Dataset.csv')
X = datasets.iloc[:, [3,4,5,6,7,8,9,10,11]].values
Y = datasets.iloc[:, 12].values



print("Real Adaboost classifier built")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)



ada_real = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME.R")
ada_real.fit(X_train, y_train)


ada_real_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
    ada_real_err[i] = zero_one_loss(y_pred, y_test)

ada_real_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
    ada_real_err_train[i] = zero_one_loss(y_pred, y_train)

#predicting using different fits


#left
#in training set
#Y_Pr=ada_real.predict([[1358552921,1358553020,	0.027450982,	0.027450982,	317	,1313	,-0.494209,	5.007181,	8.417499]])
Y_Pr=ada_real.predict([[28973453,28973531,0.333333343,0.333333343,159.85199,1497.2202,1.9259238,7.06515,6.5226417]])
#present in DC
#Y_Pr=ada_real.predict([[80428485,	80428575,	0.080078125,	0.080078125,	289.73172,	1764.2262,	1.0846461,	3.6244066,	9.04751]])
#Y_Pr=ada_real.predict([[80429621,	80429714,	0.07421875,	0.07421875,	512.52545,	1640.2806,	1.3464938,	4.077441,	8.816565]])
#not int both
#Y_Pr=ada_real.predict([[135285765,	135285964,	0.04296875,	0.04296875,	534.50507,	1934.1517,	4.202221,	4.772546,	7.4655566]])
#Y_Pr=ada_real.predict([[135286553,	135286742,	0.060546875,	0.060546875,	753.3025	,1615.2915,	5.1250362,	4.9221406,	6.758471]])

#right
#in training set
#Y_Pr=ada_real.predict([[134087519,	134087583,	0.200000018,	0.266666681,	241,	1009,	0.6772051,	7.5858974,	6.177858]])
#Y_Pr=ada_real.predict([[931656410,931656510,0.064453125,0.064453125,733.321,1608.2947,-0.9230305,1.4441271,9.655719]])
#present in DC
#Y_Pr=ada_real.predict([[80505084,	80505174,	0.078125,	0.078125,	728.3256,	1929.1539,	-0.24048555	,3.6129177,	9.113692]])
#Y_Pr=ada_real.predict([[80505504,	80505583,	0.09375,	0.09375,	562.4792,	1769.224,	-0.56075025,	3.8393779,	9.006393]])
#not in both
#Y_Pr=ada_real.predict([[135348403	,135348556	,0.04296875	,0.04296875	,736.31824	,1931.1531,	1.9641609	,4.043805	,8.715509]])
#Y_Pr=ada_real.predict([[135349215	,135349359,	0.05078125,	0.05078125,	179.8335	,1769.224,	1.7784592	,4.1834326,	8.689441]])
#
print("Predicted test results for specific user input-")
print("class :",Y_Pr)
if(Y_Pr==[0]):
    print("Typed character in left hand.")
else:
    print("Typed character in right hand.")

# Predicting the test set results
#Y_Pred = ada_discrete.predict(X_test)
Y_Pred = ada_real.predict(X_test)
#print("Predicted test results of X_test-")
#print(Y_Pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_Pred)
print("Confusion matrix is")
print(cm)

#Accuracy
from sklearn import metrics
print("Accuracy-" + str(metrics.accuracy_score(y_test,Y_Pred)))
print("--- %s seconds ---" %(time.time() - start_time))

print (time.clock() - start_time1, "seconds")


