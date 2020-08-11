# from sklearn.datasets import load_boston
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
import time
start_time = time.time()
start_time1 = time.clock()
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd


datasets = pd.read_csv('Full_Dataset.csv')
X = datasets.iloc[:,[3,4,5,6,7,8,9,10,11]].values
Y = datasets.iloc[:, 12].values


print("Voting classifier built")
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#test_size - Represents the proportion of the dataset to include in the test split.
#random_state - random_state is the seed used by the random number generator;
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

# print("Standardised data-")
# print(sc_X)
#transformation
X_Train = sc_X.fit_transform(X_Train)
# print("Transformed train data")
# print(X_Train)
X_Test = sc_X.transform(X_Test)
# print("Transformed test data")
# print(X_Test)


# Training classifiers
# clf1 = LogisticRegression(random_state=1)
# clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
# clf3 = GaussianNB()
#
#
# clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
clf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],voting='soft', weights=[2, 1, 2])


clf = clf.fit(X_Train, Y_Train)
clf.predict(X_Test[:2])

clf.score(X_Test, Y_Test)
#predicting using different fits


#left
#in training set
#Y_Pr=clf.predict([[1358552921,1358553020,	0.027450982,	0.027450982,	317	,1313	,-0.494209,	5.007181,	8.417499]])
#Y_Pr=clf.predict([[28973453,28973531,0.333333343,0.333333343,159.85199,1497.2202,1.9259238,7.06515,6.5226417]])
#present in DC
Y_Pr=clf.predict([[80428485,	80428575,	0.080078125,	0.080078125,	289.73172,	1764.2262,	1.0846461,	3.6244066,	9.04751]])
#Y_Pr=clf.predict([[80429621,	80429714,	0.07421875,	0.07421875,	512.52545,	1640.2806,	1.3464938,	4.077441,	8.816565]])
#not int both
#Y_Pr=clf.predict([[135285765,	135285964,	0.04296875,	0.04296875,	534.50507,	1934.1517,	4.202221,	4.772546,	7.4655566]])
#Y_Pr=clf.predict([[135286553,	135286742,	0.060546875,	0.060546875,	753.3025	,1615.2915,	5.1250362,	4.9221406,	6.758471]])

#right
#in training set
#Y_Pr=clf.predict([[134087519,	134087583,	0.200000018,	0.266666681,	241,	1009,	0.6772051,	7.5858974,	6.177858]])
#Y_Pr=clf.predict([[931656410,931656510,0.064453125,0.064453125,733.321,1608.2947,-0.9230305,1.4441271,9.655719]])
#present in DC
#Y_Pr=clf.predict([[80505084,	80505174,	0.078125,	0.078125,	728.3256,	1929.1539,	-0.24048555	,3.6129177,	9.113692]])
#Y_Pr=clf.predict([[80505504,	80505583,	0.09375,	0.09375,	562.4792,	1769.224,	-0.56075025,	3.8393779,	9.006393]])
#not in both
#Y_Pr=clf.predict([[135348403	,135348556	,0.04296875	,0.04296875	,736.31824	,1931.1531,	1.9641609	,4.043805	,8.715509]])
#Y_Pr=clf.predict([[135349215	,135349359,	0.05078125,	0.05078125,	179.8335	,1769.224,	1.7784592	,4.1834326,	8.689441]])
print("Predicted test results for specific user input-")
print("class :",Y_Pr)
if(Y_Pr==[0]):
    print("Typed character in left hand.")
else:
    print("Typed character in right hand.")

# Predicting the test set results
Y_Pred = clf.predict(X_Test)
#print("Predicted test results of X_test-")
#print(Y_Pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)
print("Confusion matrix is")
print(cm)

#Accuracy
from sklearn import metrics
print("Accuracy:" + str(metrics.accuracy_score(Y_Test,Y_Pred)))
print("--- %s seconds ---" %(time.time() - start_time))

print (time.clock() - start_time1, "seconds")

