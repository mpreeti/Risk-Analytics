# -*- coding: utf-8 -*-
"""
Created on Sun May 20 16:11:08 2018

@author: hp
"""

import pandas as pd
#%%Loading training n testing DS
train_data=pd.read_csv(r'C:\Users\hp\Desktop\python\risk_analytics_train.csv',header=0)
test_data=pd.read_csv(r'C:\Users\hp\Desktop\python\risk_analytics_test.csv',header=0)
#%%Preprcessing taining ds
print(train_data.shape)
train_data.head()
#%%
#finding missing values
train_data.isnull().sum()
train_data.describe(include="all")
#%%filling missing val
#inputting cat missing data with mode val

#train_data.Credit_History.mode() #cant say that person has taken loan earlier coz it say replace wid 1

colname1=['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term']
for x in colname1[:]:
    train_data[x].fillna(train_data[x].mode()[0],inplace=True) # mode is for cat...so assign index 0
train_data.isnull().sum()

#%%
#inputting numerical missing    data wid mean val
train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(),inplace=True)
print(train_data.isnull().sum())
#%%
#inputting values for credit hist column differently
train_data['Credit_History'].fillna(value=0,inplace=True)
#train_data['Credit_History']=train_data['Credit_History'].fillna(value=0)
print(train_data.isnull().sum())
#%%converting cat to num
colname=['Gender','Married','Education','Dependents','Self_Employed','Property_Area','Loan_Status']
colname
#For preprocessing the data
from sklearn import preprocessing
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()#convert cat to num by label encoding
    
for x in colname:
    train_data[x]=le[x].fit_transform(train_data.__getattr__(x))#getattr takes values  of particular column
#converted Loan status as Y-->1 and N-->0
train_data.head()
#%%Preprcessing test ds
print(test_data.shape)
test_data.head()

test_data.isnull().sum()
test_data.describe(include="all")

colname1=['Gender','Dependents','Self_Employed','Loan_Amount_Term']
for x in colname1[:]:
    test_data[x].fillna(test_data[x].mode()[0],inplace=True) # mode is for cat...so assign index 0
test_data.isnull().sum()


test_data["LoanAmount"].fillna(test_data["LoanAmount"].mean(),inplace=True)
print(test_data.isnull().sum())


test_data['Credit_History'].fillna(value=0,inplace=True)
print(test_data.isnull().sum())

colname=['Gender','Married','Education','Dependents','Self_Employed','Property_Area']
colname

from sklearn import preprocessing
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()#convert cat to num by label encoding
    
for x in colname:
    test_data[x]=le[x].fit_transform(test_data.__getattr__(x))
#Y-->1 N-->0
test_data.head()
#%%
X_train=train_data.values[:,1:-1] # 1:-1 means start from 1st col but-1 exclude last column ie loan status
Y_train=train_data.values[:,-1]
Y_train=Y_train.astype(int)
Y_train.dtype

X_test=test_data.values[:,1:] #1: means start from 1st col till last col

#%%scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler() #to avoid biasing in model
scaler.fit(X_train)
X_train=scaler.transform(X_train)
print(X_train)

scaler.fit(X_test)
X_test=scaler.transform(X_test)
print(X_test)

#%%SVM modelling
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
#from sklearn.linear_model import LogisticRegression
#svc_model=LogisticRegression()
svc_model.fit(X_train,Y_train)#Y_train is loan status
Y_pred=svc_model.predict(X_test)#using X_test we r predicting Y_pred of testing data
print(list(Y_pred))
Y_pred_col=list(Y_pred)

#%%appending predicted loan status in test data file or say copy test file to new file with name 'test data'
test_data=pd.read_csv(r'C:\Users\hp\Desktop\python\risk_analytics_test.csv',header=0)
# add new column to new test_data file
test_data["Y_predictions"]=Y_pred_col
test_data.head()


#%%
#convert test_data file into csv format with new name 'test_data1'
test_data.to_csv('test_data.csv')

# In c drive check user folder.......new test_data1 file will be there

#%%
#%%
from sklearn import cross_validation
#performing k fold cross val
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=svc_model,X=X_train,
                                                 y=Y_train,scoring="accuracy")
print(kfold_cv_result)  #gives list of 10 values of accuracy[......]

#finding mean

print(kfold_cv_result.mean())#stop in case to evaluate





#%%
#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler() #to avoid biasing in model
scaler.fit(X_train)
X_train=scaler.transform(X_train)
print(X_train)

scaler.fit(X_test)
X_test=scaler.transform(X_test)
print(X_test)
#%%
from sklearn.ensemble import GradientBoostingClassifier
model_GradientBoosting=GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model_GradientBoosting.fit(X_train,Y_train)
Y_pred=model_GradientBoosting.predict(X_test)








