import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(5,5))
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv('Breast_cancer_data.csv')
df.shape
df.head()
df.describe()

df['diagnosis'].head(5)

X = df.iloc[:, :3].values
Y = df.iloc[:, -1].values
print("X: {}".format(X.shape))
print("Y: {}".format(Y.shape))

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.175,random_state = 0)
print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("Y_train: {}".format(Y_train.shape))
print("Y_test: {}".format(Y_test.shape))

#Building our baseline dummy classifier
from sklearn.dummy import DummyClassifier
clf = DummyClassifier()
clf.fit(X_train, Y_train)
#Predicting Results
y_pred = clf.predict(X_test)
#Calculating Resulta
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print("CM: \n",confusion_matrix(Y_test, y_pred))
print("acc: {0}%".format(accuracy_score(Y_test, y_pred) * 100))

#Random Forest Classifier
st=dt.now()
randomforest = RandomForestClassifier(n_estimators = 100,random_state = 0)
randomforest.fit(X_train, Y_train)
print("Time taken to complete random search: ",dt.now()-st)
random_pred = randomforest.predict(X_test)
#Model Evaluation
rmacc = accuracy_score(Y_test, random_pred)
print('Accuracy Score: ' + str(rmacc))

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print('Precision Score: ' + str(precision_score(Y_test, random_pred)))
print('Recall Score: ' + str(recall_score(Y_test, random_pred)))
print('F1 Score: ' + str(f1_score(Y_test, random_pred)))

#Random Forest Model
import pickle
filename = 'Breast_Cancer.sav'
pickle.dump(randomforest, open(filename, 'wb'))