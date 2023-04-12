import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# setting path
import sys
sys.path.append('../../')
import utils

"""
 the preprocess_data function is an important step in the machine learning pipeline that helps
 ensure that the data is in the right format and on the same scale before training the model.
"""
def preprocess_data(X, y):
    """
    Preprocess the given features and target variable and return the training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    """
    Split the data into training and testing sets: This is done using the train_test_split function from scikit-learn.
    The test_size parameter is set to 0.2, which means that 20% of the data is used for testing,
    and the random_state parameter is set to 42 to ensure that the data is split in the same way every time the function is called.
    """
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    """
    Standardize the data: This is done using the StandardScaler class from scikit-learn. 
    Standardization is a common preprocessing step that scales the data to have zero mean and unit variance. 
    This can help improve the performance of the model by ensuring that all the features are on the same scale. 
    The fit_transform method is called on the training data, and the transform method is called on the testing data. 
    This ensures that the mean and variance of the training data are used to scale the testing data, which helps prevent information leakage from the testing set.
    """

    return X_train, X_test, y_train, y_test
    """
    Return the preprocessed data: The function returns four variables: X_train, X_test, y_train, and y_test. 
    X_train and y_train are the preprocessed training features and target variable, and X_test and y_test are the preprocessed testing features and target variable. 
    These variables are used in the train_model and evaluate_model functions to train and evaluate the logistic regression model.
    """

def train_model(X_train, y_train):
    """
    Train a logistic regression model on the given training data and return the trained model.
    """
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the given model on the testing data and print the accuracy score and confusion matrix.
    """
    y_pred = model.predict(X_test)
    print("Accuracy score:", accuracy_score(y_test, y_pred) * 100, "%")
    utils.print_confusion_matrix(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('./data/diabetes.csv')

    # Split the dataset into features and target variable
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    """
    X, X_train contains all the input variables (features) for the training set, 
    except for the Outcome column. y, y_train, on the other hand, contains only the Outcome column for the training set. 
    This allows us to train the logistic regression model to predict whether a patient has diabetes or not based on the other input variables (features).
    """
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    utils.save_model(model,"./model/my_model.pkl")

