import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import random  
from utils.data_visualizations import visualize_unique_counts
class DataLoader:
    def __init__(self, data_path,target_column_name, test_size=0.2):
        self.train_data = None
        self.test_data = None
        self.data = None
        self.target_column_name = target_column_name
        self.data_path = data_path

        self.load_data()
        self.train_data, self.test_data = self.test_train_split(test_size=test_size)

    def load_data(self):
        # Loading entire data from csv
        with open(self.data_path, "r") as file:
            self.data = pd.read_csv(file,na_values=['?'])
        return self.data
    
    def test_train_split(self, test_size=0.2):
        # Split the data into train, test and validation sets
        X = self.data.drop(self.target_column_name, axis=1) 
        y = self.data[self.target_column_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random.randint(0, 2**32 - 1))
        self.train_data = pd.concat([X_train, y_train], axis=1)
        self.test_data = pd.concat([X_test, y_test], axis=1)
        return self.train_data, self.test_data
    
    def generate_plots(self):
        columns_to_plot = ["gender", "race", "age"]
        for col in columns_to_plot:
            visualize_unique_counts(self.data, col, save_path=f"/Users/manu/Desktop/Projects/XAI-Diabetes-Readmission/Outputs/{col}_distribution.png")