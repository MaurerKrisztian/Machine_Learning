import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.metrics import confusion_matrix

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

def print_confusion_matrix(cm):
    TP = cm[1, 1]
    FP = cm[0, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    print("True Positives:", TP)
    print("False Positives:", FP)
    print("True Negatives:", TN)
    print("False Negatives:", FN)