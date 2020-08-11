# Random Forest Classifier

# Importing the libraries
import time
start_time = time.time()
start_time1 = time.clock()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# Importing the datasets

datasets = pd.read_csv('Keystrokes (2).csv')
feature_names=['DownTime, UpTime, Pressure, FingerArea, RawX, RawY, gravityX, gravityY, gravityZ']
target_names=['Hands']
X = datasets.iloc[:,[3,4,5,6,7,8,9,10,11]].values
# print("X")
# print(X)
Y = datasets.iloc[:, 12].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# print("X_Train-")
# print(X_Train)
# print("X_Test-")
# print(X_Test)
# print("Y_Train-")
# print(Y_Train)
# print("Y_Test-")
# print(Y_Test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# print("Standardised data-")
# print(sc_X)

X_Train = sc_X.fit_transform(X_Train)
# print("Transformed train data")
# print(X_Train)

X_Test = sc_X.transform(X_Test)
# print("Transformed test data")
# print(X_Test)


# Fitting the classifier into the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1, criterion = 'entropy', random_state = 0)
classifier.fit(X_Train,Y_Train)
print("RF classifier built for the initial dataset! ")


# Predicting the test set results for user input

# #1. left
#Y_Pr = classifier.predict([[1358552921,1358553020,0.027450982,0.027450982,317,1313,-0.494209,5.007181,8.417499]])
# #2. right
#Y_Pr = classifier.predict([[134087519,134087583,0.200000018,0.266666681,241,1009,0.6772051,7.5858974,6.177858]])
# #3. left
#Y_Pr = classifier.predict([[80428485,80428575,0.080078125,0.080078125,289.73172,1764.2262,1.0846461,3.6244066,9.04751]])
# #4. left
#Y_Pr = classifier.predict([[80429621,80429714,0.07421875,0.07421875,512.52545,1640.2806,1.3464938,4.077441,8.816565]])
# #5.right
#Y_Pr = classifier.predict([[80505084,80505174,0.078125,0.078125,728.3256,1929.1539,-0.24048555,3.6129177,9.113692]])
# #6. right
#Y_Pr = classifier.predict([[80505504,80505583,0.09375,0.09375,562.4792,1769.224,-0.56075025,3.8393779,9.006393]])
# #7. left
#Y_Pr = classifier.predict([[135285765,135285964,0.04296875,0.04296875,534.50507,1934.1517,4.202221, 4.772546,7.4655566]])
# #8. left
#Y_Pr = classifier.predict([[135286553,135286742,0.060546875,0.060546875,753.3025,1615.2915,5.1250362,4.9221406,6.758471]])
# #9.right
Y_Pr = classifier.predict([[135348403,135348556,0.04296875,0.04296875,736.31824,1931.1531,1.9641609,4.043805,8.715509]])
# #10. right
#Y_Pr = classifier.predict([[135349215,135349359,0.05078125,0.05078125,179.8335,1769.224,1.7784592,4.1834326,8.689441]])

#112, MMB29P, 0, h, 65472653, 65472729, 0.49803924560546875, 0.22580644488334656, 428.405, 1015.20685, 0.45893854, 4.8567877, 8.507048, 1, 0, 0
#Y_Pr = classifier.predict([[65472653, 65472729, 0.49803924560546875, 0.22580644488334656, 428.405, 1015.20685, 0.45893854, 4.8567877, 8.507048]])
#112, MMB29P, 0, i, 65472866, 65472957, 0.4901961088180542, 0.19354838132858276, 518.28015, 905.2927, 0.4269161, 4.931457, 8.465598, 1, 0, 0
#Y_Pr = classifier.predict([[65472866, 65472957, 0.4901961088180542, 0.19354838132858276, 518.28015, 905.2927, 0.4269161, 4.931457, 8.465598,]])
#112, MMB29P, 0, Space, 65473330, 65473421, 0.49803924560546875, 0.19354838132858276, 403.43967, 1245.0273, 0.45070848, 4.9465704, 8.455572, 1, 0, 0
#Y_Pr = classifier.predict([[65473330, 65473421, 0.49803924560546875, 0.19354838132858276, 403.43967, 1245.0273, 0.45070848, 4.9465704, 8.455572]])
#112, MMB29P, 0, s, 65473876, 65473949, 0.49803924560546875, 0.22580644488334656, 173.75867, 1014.20764, -0.25573066, 5.5898623, 8.053347, 1, 0, 0
#Y_Pr = classifier.predict([[65473876, 65473949, 0.49803924560546875, 0.22580644488334656, 173.75867, 1014.20764, -0.25573066, 5.5898623, 8.053347]])
#112, MMB29P, 0, o, 65474468, 65474540, 0.250980406999588, 0.16129031777381897, 561.2205, 909.2896, 0.32396543, 5.1393037, 8.345738, 1, 0, 0
#Y_Pr = classifier.predict([[65474468, 65474540, 0.250980406999588, 0.16129031777381897, 561.2205, 909.2896, 0.32396543, 5.1393037, 8.345738]])
#1126, MMB29P, 0, u, 65474896, 65474977, 0.49803924560546875, 0.19354838132858276, 450.37448, 917.2834, 0.33728316, 5.258565, 8.270471, 1, 0, 0
#Y_Pr = classifier.predict([[65474896, 65474977, 0.49803924560546875, 0.19354838132858276, 450.37448, 917.2834, 0.33728316, 5.258565, 8.270471]])
#11#2, MMB29P, 0, m, 65475260, 65475359, 0.250980406999588, 0.16129031777381897, 552.23303, 1146.1046, 0.2599205, 5.1506763, 8.34095, 1, 0, 0
#Y_Pr = classifier.predict([[65475260, 65475359, 0.250980406999588, 0.16129031777381897, 552.23303, 1146.1046, 0.2599205, 5.1506763, 8.34095]])
#11#2, MMB29P, 0, y, 65475879, 65475987, 0.49803924560546875, 0.22580644488334656, 383.4674, 924.2779, 0.0664391, 5.207688, 8.309227, 1, 0, 0
#Y_Pr = classifier.predict([[65475879, 65475987, 0.49803924560546875, 0.22580644488334656, 383.4674, 924.2779, 0.0664391, 5.207688, 8.309227]])
#112, MMB29P, 0, a, 65476934, 65477007, 0.49803924560546875, 0.3870967626571655, 90.87379, 1003.21625, -0.05431845, 5.2139726, 8.305486, 1, 0, 0
#Y_Pr = classifier.predict([[65476934, 65477007, 0.49803924560546875, 0.3870967626571655, 90.87379, 1003.21625, -0.05431845, 5.2139726, 8.305486]])


print("Predicted test results for specific user input-")
print("class :", Y_Pr)
if(Y_Pr==[0]):
    print("Typed character in left hand.")
else:
    print("Typed character in right hand.")

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)
# print("Predicted test results of X_test-")
# print(Y_Pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)
print("Confusion matrix is")
print(cm)

#Accuracy
from sklearn import metrics
print("Accuracy-" + str(metrics.accuracy_score(Y_Test,Y_Pred)))
print("---The time taken for execution is %s seconds ---" %(time.time() - start_time))

# Visualising the Training set results
#,'blue','yellow','black','brown','gold','purple','pink'
from matplotlib.colors import ListedColormap
# X_Set, Y_Set = X_Train, Y_Train
# X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))

# print(X1.ravel())
# print(X2.ravel())
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red','green')))

# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(Y_Set)):
#     plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Random Forest Classifier (Training set)')
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.legend()
# plt.show()

# #Extract single tree
# estimator = classifier.estimators_[0]
#
# from sklearn.tree import export_graphviz
# # Export as dot file
# export_graphviz(estimator, out_file='tree2.dot',
#                 rounded = True, proportion = False,feature_names = classifier.feature_importances_,
#                 class_names = classifier.target_importances_,
#                 precision = 2, filled = True)
#
# # Convert to png using system command (requires Graphviz)
# from subprocess import call
# call(['dot', '-Tpng', 'tree2.dot', '-o', 'tree2.png', '-Gdpi=600'])

# Calculate feature importances
# importances = classifier.feature_importances_
# # Sort feature importances in descending order
# indices = np.argsort(importances)[::-1]
#
# # Rearrange feature names so they match the sorted feature importances
# names = ['DownTime', 'UpTime', 'Pressure', 'FingerArea', 'RawX', 'RawY', 'gravityX', 'gravityY', 'gravityZ']
#
# # Barplot: Add bars
# plt.bar(range(X.shape[1]), importances[indices])
# # Add feature names as x-axis labels
# plt.xticks(range(X.shape[1]), names, rotation=20, fontsize = 8)
# # Create plot title
# plt.title("Feature Importance")
# # Show plot
# plt.show()
