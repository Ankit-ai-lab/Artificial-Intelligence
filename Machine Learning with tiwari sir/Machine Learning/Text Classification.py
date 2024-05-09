# reading data

import pandas as pd
import os

os.chdir(r"C:\Users\PraveshTiwari\OneDrive - TheMathCompany Private Limited\Documents\Python Scripts\NLP")

data = pd.read_csv('spam.csv', encoding = 'latin1')

data.head()

# drop unnecessary columns and rename cols
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

data.columns = ['label', 'text']

data.head()

# check missing values
data.isna().sum()

# check data shape

data.shape

# check target balance

data['label'].value_counts(normalize = True).plot.bar()

# text preprocessing

# download nltk

import nltk

#nltk.download('all')

# create a list text

text = list(data['text'])

# preprocessing loop

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    r = re.sub('[^a-zA-Z]', ' ', text[i])
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words('english')]
    #r = [lemmatizer.lemmatize(word) for word in r]
    r = ' '.join(r)
    corpus.append(r)

#assign corpus to data['text']

data['text'] = corpus
data.head()

# Create Feature and Label sets

X = data['text']
y = data['label']

# train test split (66% train - 33% test)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

print('Training Data :', X_train.shape)
print('Testing Data : ', X_test.shape)


# Feature Extraction

# Train Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train_cv = cv.fit_transform(X_train)

X_train_cv.shape

# Training Logistic Regression model

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train_cv, y_train)

# transform X_test using CV

X_test_cv = cv.transform(X_test)

# generate predictions

predictions = lr.predict(X_test_cv)

predictions

import pandas as pd

from sklearn import metrics

df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])

df
