#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:33:29 2022

@author: aiturgan
"""

#1.DATA EXPLORATION

import pandas as pd
import matplotlib.pyplot as plt

df_disease = pd.read_csv('Data.csv')
df_disease.drop('Unnamed: 133', axis=1, inplace=True)
print(df_disease)
print(df_disease.columns.values) #printing column names
print(df_disease.dtypes) #printing data type of columns


print(df_disease.describe())#description
print(df_disease.head(3))#first three records



#2. handling missing values

#check for null values in df
print(df_disease.isnull().sum()) #sum of null values in each column
percent_missing = df_disease.isnull().sum() * 100 / len(df_disease) 
print(df_disease.isnull().sum()) #sum of null values in each column


#look for categorical data
categoricals = []
for col, col_type in df_disease.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     
print(categoricals)#1 categoricals


#HANDLING COLUMNS
#there are two column with similar  names fluid_overload and fluid_overload.1

df_disease[df_disease['fluid_overload'] == 1]

df_disease[df_disease['fluid_overload.1'] == 1].prognosis.unique()

#looking for symptoms and their prognosis that mostly telling the disease

data = {'Symptoms': [], 'Prognosis': [], 'length': []}
table = pd.DataFrame(data)
table = table.astype({"Symptoms": str, "Prognosis": object, 'length': int})
i = 0

for symp in sorted(df_disease.columns.tolist()[:-1]):
    prognosis = df_disease[df_disease[symp] == 1].prognosis.unique().tolist()
    table = table.append({'Symptoms': symp}, {'Prognosis': prognosis}, {'length':len(prognosis)}) 
    table.at[i,'Prognosis'] = prognosis
    table.at[i, 'length'] = len(prognosis)
    i += 1
    
table.sort_values(by='length', ascending=False).head(10)
#So, fatigue and vomiting are the two most common and most generic symptoms in 
#this dataset and probably won't be an unique/significant predictor for an illness

#What are most telling symptoms
table.sort_values(by='length', ascending=True).head(10)

#As observed above, fluid_overload does seem to be a problematic symptom for our dataset. 
#Apart from that, there are a great number of very telling symptoms such as patches_in_throat,
# with the prognosis being AIDS.



#CHANGING TARGET FEATURE TO NUMERICAL FOR MODEL TO UNDERSTAND IT

features = df_disease.iloc[:,0:-1] #SEPARATING TARGET FEATURE FROM OTHER COLUMNS
target = df_disease.prognosis #IDENTIFYING PROGNOSIS COLUMN AS TARGET FEATURE

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(target.tolist())

encoded_target = le.transform(target)

print(encoded_target) #each column has unique encoded number

#dividing data into testing and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, encoded_target, test_size=0.20, random_state=0)


#Model Implementation
# Support Vector Classification

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

svc_model = SVC()

svc_model.fit(X_train,y_train)

pred = svc_model.predict(X_test)

score = accuracy_score(y_test, pred)
print("Testing Accuracy score for SVC is {}%".format(score * 100))


# KNN (k-nearest neighbors) model

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train, y_train)

pred = knn_model.predict(X_test)

score = accuracy_score(y_test, pred)
print("Accuracy score for KNN is {}%".format(score*100))


import joblib 
joblib.dump(svc_model, 'svc_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')
print("Models dumped!")

model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Model columns dumped!")