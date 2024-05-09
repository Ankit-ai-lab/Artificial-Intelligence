from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
print("imported!")

os.chdir(r"C:\Users\PraveshTiwari\OneDrive - TheMathCompany Private Limited\Documents\Python Scripts\My docs")

data = pd.read_csv("Titanic.csv")

data["Age"] = data["Age"].fillna(round(data["Age"].mean(),0))
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

data.head()

missing_data = data[data.Embarked.isnull()]

# Drop the unnecessary columns
data_model = data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis='columns')
data_model.head()

# Dealing with missing values
mv = data_model.isnull().sum()
mv


data_with_dummies = pd.get_dummies(data_model, prefix = ["Sex", "Embarked"])
data_with_dummies.head()

Y=data_with_dummies.Survived
X=data_with_dummies
X.drop(['Survived'],axis=1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


classifier= GaussianNB()
classifier.fit(X_train, y_train)
classifier.class_prior_

predicts=classifier.predict(X_test)
accuracy=round(accuracy_score(predicts,y_test),3)
print(accuracy)