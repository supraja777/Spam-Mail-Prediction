import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

raw_mail_data = pd.read_csv('mail_data.csv')
print(raw_mail_data)

raw_mail_data.describe()

null_counts = raw_mail_data.isnull().sum()
print(null_counts)

raw_mail_data.head()

"""Label Encoding - Spam = 0, Ham = 1


raw_mail_data.loc[...]: The .loc indexer is used to access a group of rows and columns by label(s) or a boolean array. In this case, it selects rows where the condition raw_mail_data['Category'] == 'spam' is True.

... , 'Category'] = 0: This part assigns the value 0 to the 'Category' column for the selected rows. So, wherever the condition raw_mail_data['Category'] == 'spam' is True, the 'Category' column for those rows will be set to 0.
"""

raw_mail_data.loc[raw_mail_data['Category'] == 'spam', 'Category'] = 0
raw_mail_data.loc[raw_mail_data['Category'] == 'ham', 'Category'] = 1

print(raw_mail_data)

#separating labels and data
X = raw_mail_data['Message']
Y = raw_mail_data['Category']
print(X)
print(Y)

#Splitting data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3)
print("X train")
print (X_train)
print()
print("X test")
print (X_test)
print()
print("Y train")
print (Y_train)
print()
print("Y test")
print (Y_test)

print(X_train.shape)
print(Y_train.shape)

"""Feature Extraction

"""

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

print(X_train_features)
print(X_test_features)

#converting y_train and y_test into integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(Y_train)
print(Y_test)

"""Training the model using logistic regression

"""

model = LogisticRegression()
#training the logistic regression
model.fit(X_train_features, Y_train)

#prediction on training data
training_data_prediction = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, training_data_prediction)

print("Accuracy on training data == ", accuracy_on_training_data)

#prediction on test data
test_data_prediction = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, test_data_prediction)

print("Accuracy on test data == ", accuracy_on_test_data)
