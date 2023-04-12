import pandas as pd
import seaborn as sns

# Perform exploratory data analysis
def explore_data():
    """
    Load the diabetes dataset from a CSV file and return a pandas dataframe.
    """
    df = pd.read_csv("./data/diabetes.csv")

    """
    Perform exploratory data analysis on the given dataframe.
    """
    print("Head of the dataset:\n", df.head())
    print("Shape of the dataset:", df.shape)
    print("Data types of the columns:\n", df.dtypes)
    print("Summary statistics of the dataset:\n", df.describe())
    sns.countplot(x='Outcome', data=df)

explore_data()