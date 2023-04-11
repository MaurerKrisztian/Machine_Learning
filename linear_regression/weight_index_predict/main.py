import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
import utils


def make_predictions(model, height, weight):
    new_data = np.array([[height, weight]])
    predictions = model.predict(new_data)
    print(f"New data: {new_data} --> predicted index: {predictions} {get_category(round(predictions[0]))}")
    return predictions[0]

def get_category(index):
    map = {
        0: 'Extremely Weak',
        1: 'Weak',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obesity',
        5: 'Extreme Obesity'
    }
    return map.get(index, 'Invalid Index')
    

def main():
    model = utils.load_model("./model/my_model.pkl")
    height=177
    weight=76
    make_predictions(model,  height, weight)

if __name__ == '__main__':
    main()