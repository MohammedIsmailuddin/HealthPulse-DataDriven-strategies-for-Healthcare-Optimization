import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load training data
train_data = pd.read_csv("C:\\Users\\91800\\Desktop\\Health care\\data\\raw\\train_data.csv")
test_data = pd.read_csv("C:\\Users\\91800\\Desktop\\Health care\\data\\sample_sub.csv")

# Data preprocessing
# Example: Fill missing values with median and encode categorical features
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Encoding categorical features (Example: One-hot encoding)
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

# Align test data columns to train data
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# Split train data into features and target
X = train_data.drop(['Stay', 'case_id'], axis=1)
y = train_data['Stay']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

# Model training
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Model evaluation
y_pred = rf.predict(X_val)
print(classification_report(y_val, y_pred))

# Predict on test data
test_predictions = rf.predict(test_data)

# Prepare submission file
submission = pd.DataFrame({
    'case_id': test_data['case_id'],
    'Stay': test_predictions
}, index=test_data.index)

submission.to_csv('C:\\Users\\91800\\Desktop\\Health care\\data\\submission.csv', index=False)