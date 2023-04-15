import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle


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
    
def load_model(file_path):
    print(f"Load model from: {file_path}")
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model
    
def main(args):
    model = load_model(args.model_path)
    height = args.height
    weight = args.weight

    if height is None or weight is None or np.isnan(height) or np.isnan(weight):
        print("Error: Invalid input. Height and weight must be numeric values.")
        return

    make_predictions(model, height, weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict BMI category based on height and weight.')
    parser.add_argument('--height', type=int, help='height in centimeters')
    parser.add_argument('--weight', type=int, help='weight in kilograms')
    parser.add_argument('--model-path', type=str, default='./model.pkl', help='path to the trained model file')
    args = parser.parse_args()
    main(args)
