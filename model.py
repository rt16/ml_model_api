# Import dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
dataset_url = "home/aarti/Desktop/msc-2/mental_health_flask\stress_data_set.csv"
col_names = ['I found it hard to wind down',
                 'I tended to over-react to situations',
                 'I felt that I was using a lot of nervous energy',
                 'I found myself getting agitated',
                 'I found it difficult to relax',
                 'I was intolerant of anything that kept me from getting on with what I was doing',
                 'I felt that I was rather touchy',
                 'Result']
data = pd.read_csv(dataset_url)
print(data);
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
#apply label encoding
data['I found it hard to wind down'] = label_encoder.fit_transform(data['I found it hard to wind down'])
data['I tended to over-react to situations'] = label_encoder.fit_transform(data['I tended to over-react to situations'])
data['I felt that I was using a lot of nervous energy'] = label_encoder.fit_transform(data['I felt that I was using a lot of nervous energy'])
data['I found myself getting agitated'] = label_encoder.fit_transform(data['I found myself getting agitated'])
data['I found it difficult to relax'] = label_encoder.fit_transform(data['I found it difficult to relax'])
data['I was intolerant of anything that kept me from getting on with what I was doing'] =label_encoder.fit_transform(data['I was intolerant of anything that kept me from getting on with what I was doing'])
data['I felt that I was rather touchy'] = label_encoder.fit_transform(data['I felt that I was rather touchy'])
data['Result'] = label_encoder.fit_transform(data['Result'])
print(data.head(20))
feature_cols = ['I found it hard to wind down',
                    'I tended to over-react to situations',
                    'I felt that I was using a lot of nervous energy',
                    'I found myself getting agitated',
                    'I found it difficult to relax',
                    'I was intolerant of anything that kept me from getting on with what I was doing',
                    'I felt that I was rather touchy']
result_cols = ['Result']
X = data[feature_cols]  # Features
y = data[result_cols]  # Target variable
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)  # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)
print(clf.predict([[1,1,1,1,1,1,1]]))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Save your model
import joblib
joblib.dump(clf, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
clf = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(X.columns)
print(model_columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
