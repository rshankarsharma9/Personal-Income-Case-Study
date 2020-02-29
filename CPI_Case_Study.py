# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:49:20 2020

@author: Ravishankar
"""
###############################################
#Classifying Personal Income Case Study
###############################################

#To work with Data Frames
import pandas as pd
#To perform numerical operations
import numpy as np
#To visualize Data
import matplotlib.pyplot as plt
import seaborn as sns
#To partition the data
from sklearn.model_selection import train_test_split
#Importing library for logistic regression
from sklearn.linear_model import LogisticRegression
#Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

#################
#Importing Data 
#################
data_income=pd.read_csv('income.csv')
 
 #creating a copy of original data
data=data_income.copy()
 
"""
 #Exploratory data analysis:
 #1. Getting to know the data
 #2. Data preprocessing (missing Values)
 #3. Cross table and data visualization
"""
 ###########################
 #Getting to know the data
 ###########################
     
 #To check the variables' data type 
print(data.info())
 # Checking for missing values
print('Data columns with null values:\n',data.isnull().sum())
 
 # No missing Values !
# Summary of numerical variables
summary_num=data.describe()
print(summary_num)
# Summary of categorical variables
summary_cate=data.describe(include="O")
print(summary_cate)
 
#***Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()
#note: are are some missing values in form of missing values ???

#checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
 
#There exists '?' instend of nan 
"""
Go back and read the data by including "na_values[' ?'] to consider ' ?' as nan
"""
data=pd.read_csv('income.csv',na_values=[" ?"])
###########################
#data pre-processing
###########################

data.isnull().sum()
# Now you can see there are some missing values available in jobtype and occupation

missing =data[data.isnull().any(axis=1)]
#axis=1 =>to consider at least one column value is missing
""" point to be note:
1. Missing values for jobtype =1809
2. Missing values for occupation =1816
3. There are 1809 rows where two specific columns i.e occupation and jobtype 
have missing values
4.(1816-1809)=7 => you stil have occupation unfilled for these 7 rows.
because jpb type is neverworked 
"""
# Herewe remove all the rows that have missing values
data2=data.dropna(axis=0)

#Relationship between independent variables
correlation = data2.corr()

######################################
#Cross table & Data visualization
#####################################

#Extracting the column names
data2.columns

######################################
#Gender proportion table: 
#####################################
gender=pd.crosstab(index=data2['gender'],columns='count',normalize=True)
print(gender)

######################################
#Gender vs Salary status:
#####################################
gender_salstat=pd.crosstab(index=data2['gender'],columns=data2['SalStat'],normalize='index',margins=True)
print(gender_salstat)

#########################################
#Frequency distribution of salary status
#########################################

SalStat=sns.countplot(data2['SalStat'])
"""
Note: 75% salary Status<=50000 and 25% salary status>50000
"""

###########Histogram of Age###########################
sns.distplot(data2['age'],bins=10,kde=False)
"""
Note:people with age 25-40 age are high in frequency
"""

###########Box Plot- Age vs Salary Status ##################
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()
"""
Note: People with age 35-50 age salary status is >50000 
and 25-35 age salary status <=50000
"""
###########Bar Plot - job type vs Salary status
sns.countplot(y='JobType',hue='SalStat',data=data2)

jobtype_salstat=pd.crosstab(index=data2['JobType'],columns=data2['SalStat'],normalize='index')
print(jobtype_salstat)

"""
Note: 56% of self-emp-inc jobtype people salary status >50000
so we can avoid these people
"""  
##########Bar Plot - Education vs Salary status
sns.countplot(y='EdType',hue='SalStat',data=data2)

jobtype_salstat=pd.crosstab(index=data2['EdType'],columns=data2['SalStat'],normalize='index')
print(jobtype_salstat)

"""
Note:people with doctorate,prof-school and masters are more 
likely to earn above 50000 so we can avoid these people
"""
##########Bar Plot - occupation vs Salary status
sns.countplot(y='occupation',hue='SalStat',data=data2)

jobtype_salstat=pd.crosstab(index=data2['occupation'],columns=data2['SalStat'],normalize='index')
print(jobtype_salstat)
"""
Note:people with exec-managerial and prof-specialty are more 
likely to earn above 50000 so we can avoid these people
"""
##########Histogram - Capital gain
sns.distplot(data2['capitalgain'],kde=False)

"""
Note: almost 92 % people with zero capital gain
"""
##########Histogram - Capital Loss
sns.distplot(data2['capitalloss'],kde=False)

"""
Note: almost 95 % people with zero capital loss
"""
##########Boxplot - hour per week vs Salary status

sns.boxplot('SalStat','hoursperweek',data=data2)
data2.groupby('SalStat')['hoursperweek'].median()
"""
Note: people who work for 40-50 hr per weeks are make more than 50000
"""

#############################################
#Logistic Regression
############################################

#Reindexing the Salary Status names to 0 ,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

#Here we convert cetegorical variable to numerical value either 0 or 1
new_data=pd.get_dummies(data2,drop_first=True)

#Storing the columns names
columns_list=list(new_data.columns)
print(columns_list)

#Seprating Input names from data

features=list(set(columns_list)-set(['SalStat']))
print(features)

#storing output values in Y
y=new_data['SalStat'].values
print(y)

#Storing the values from input features
x=new_data[features].values
print(x)

#Spliting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0)

#Make a instance of model
logistic=LogisticRegression()

#Fitting the values for x and Y
logistic.fit(train_x,train_y)

#Predict from test Data
prediction=logistic.predict(test_x)
print(prediction)

#confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

#Calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

##########################################################
################## KNN- K Nearest Neighbors classifier
##########################################################

#importing the KNN library
from sklearn.neighbors import KNeighborsClassifier

#Make a instance of model
KNN_classifier=KNeighborsClassifier(n_neighbors=5)

#Fitting the values for x and Y
KNN_classifier.fit(train_x,train_y)

#Predict from test Data
prediction=KNN_classifier.predict(test_x)
print(prediction)

#confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
confusion_matrix

#Calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
accuracy_score

print('Misclassified sample : %d' % (test_y!=prediction).sum())

"""
Effect of K  value on classifier
"""

Misclassified_sample=[]
#calculating error for k values b/w 1 to 20
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y!=pred_i).sum())
    
print(Misclassified_sample)


##################################################
#####################END- Script
##################################################








