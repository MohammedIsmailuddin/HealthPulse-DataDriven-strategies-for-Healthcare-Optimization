#1.Load Data

import pandas as pd 
# Load the dataset
data = pd. read_csv("C:\\Users\\91800\\Desktop\\Health care\\data\\test_data.csv")
print  (data.head())


#2.Data preprocessing
#basic data preprocessing 
#handling missing values
data.fillna(method='ffill', inplace=True) 

#convert categorical columns to numerical
data =pd.get_dummies(data, drop_first=True)

print(data.info())


#3.Feature Engineering

import pandas as pd
from sklearn.model_selection import train_test_split

# Let's assume 'Age', 'patient id', 'Type of Admission' are features
# and 'case_id' is the target variable.

# Selecting features and target variable
features = data[['Age', 'patientid', 'Type of Admission']]
target = data['case_id']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

#4.model training
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

#5.model evalution
from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')