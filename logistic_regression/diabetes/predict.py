import sys
sys.path.append('../../')
import utils
import pandas as pd


loaded_model = utils.load_model("./model/my_model.pkl")
new_data = pd.DataFrame({'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72, 'SkinThickness': 35, 
                        'Insulin': 0, 'BMI': 33.6, 'DiabetesPedigreeFunction': 0.627, 'Age': 50},
                    index=[0])
new_prediction = loaded_model.predict(new_data)
print("Prediction for: \n ", new_data)
print("(1) you got diabetes, (0) you don't got diabetes ")
print("Result: ",new_prediction)