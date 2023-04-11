import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def save_model(model, file_path):
    print(f"save the model to: {file_path}")
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_path):
    print(f"Load model from: {file_path}")
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model
