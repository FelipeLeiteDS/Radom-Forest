# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:21:43 2023

@author: Felipe Leite
"""

from pandas import read_csv
df=read_csv("C:/Users/Felipe Leite/OneDrive/Área de Trabalho/Master UCW/Fourth Term/BUSI 652 - Predictive Analysis/2nd group assignment/dataset_true/dataset_true/heart.csv")

y=df["output"]
x=df.drop("output", axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier as RFR

model_rf=RFR(n_estimators=5000,max_features=3,max_depth=8)
model_rf.fit(x_train,y_train)
y_pred=model_rf.predict(x_test)

model_rf.predict_proba(x_test)

y_pred=(model_rf.predict(x_test)>0.5).astype("int32")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
accuracy_score(y_test, y_pred)

print(y_pred)

model_rf.feature_importances_

print(classification_report(y_test, y_pred))

#For Random Forest, we can use 1,2,3,4... as dummy variables
