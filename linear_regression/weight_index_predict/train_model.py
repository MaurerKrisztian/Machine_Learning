import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

def get_data(path):
    print(f"Get data from {path}")
    data = pd.read_csv(path)
    X = data[['Height', 'Weight']]
    y = data['Index']
    return X, y


def train_model(x, y):
    print(f"Train the model..")

    regression_model = LinearRegression()
    regression_model.fit(x, y)
    return regression_model

def evaluate_model(model, x, y):
    print(f"Evaluate the model..")
    r_squared = model.score(x, y)
    print('R-squared:', r_squared)

def main():
    x, y = get_data("./data/peoples.csv")
    model = train_model(x, y)
    evaluate_model(model, x, y)
    utils.save_model(model, './model/my_model.pkl')


if __name__ == '__main__':
    main()
