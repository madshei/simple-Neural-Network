import pandas as pd
from sklearn.model_selection import train_test_split
from Accuracy import accuracy
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)

# Load the dataset
data = pd.read_csv('diabetes_pre.csv')

# gender_dict = {'Female': 0, 'Male': 1}

# data['gender'] = data.gender.map(gender_dict)

# Split the dataset into features and target variable
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]


# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# send data to neural network module
accuracy(X, X_train, X_test, y_train, y_test)
