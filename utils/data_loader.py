import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
            self.data = pd.read_csv(file,na_values=['?','None'])
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

    def preprocess_data(self, inplace=True):
        """
        Preprocess the data by performing the following steps:
        1. Drop irrelevant columns.
        2. Handle missing values by dropping rows with missing values in critical columns.
        3. Convert categorical variables to numerical using one-hot encoding.
        4. Scale numerical features using StandardScaler.
        """
        
        process_data = self.data.copy()  


        process_data.drop_duplicates(subset='patient_nbr', keep='first', inplace=True)


        columns_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'A1Cresult', 'max_glu_serum']
        process_data = process_data.drop(columns=columns_to_drop)


        expired_ids = [11, 13, 14, 19, 20, 21]
        process_data = process_data[~process_data['discharge_disposition_id'].isin([str(i) for i in expired_ids])]  #Expried patients, pointless for readmission prediction


        process_data['readmitted'] = process_data['readmitted'].apply(lambda x: 1 if x == '<30' else 0)


        process_data.drop(columns=['examide', 'citoglipton', 'glimepiride-pioglitazone'], inplace=True)


        def lambda_func(x):
            x = x[1:-1].split('-')
            if len(x) > 1:
                return (int(x[0]) + int(x[1])) // 2
            else:
                raise ValueError("Invalid age format")
        process_data['age'] = process_data['age'].map(lambda_func)


        #check this!
        def map_icd9(code):
            try:
                code = float(code)
                if 390 <= code <= 459 or code == 785: return 'Circulatory'
                elif 460 <= code <= 519 or code == 786: return 'Respiratory'
                elif 520 <= code <= 579 or code == 787: return 'Digestive'
                elif code == 250: return 'Diabetes'
                elif 800 <= code <= 999: return 'Injury'
                elif 710 <= code <= 739: return 'Musculoskeletal'
                elif 580 <= code <= 629 or code == 788: return 'Genitourinary'
                else: return 'Other'
            except:
                return 'Other'
        for col in ['diag_1', 'diag_2', 'diag_3']:
            process_data[col] = process_data[col].apply(map_icd9)


        process_data['admission_type_id'] = process_data['admission_type_id'].astype(str)
        process_data['discharge_disposition_id'] = process_data['discharge_disposition_id'].astype(str)
        process_data['admission_source_id'] = process_data['admission_source_id'].astype(str)


        columns_to_check = ['race']
        process_data = process_data.dropna(subset=columns_to_check)
        
        #scaler after split
        # scaler = StandardScaler()
        # process_data[self.numerical_columns] = scaler.fit_transform(process_data[self.numerical_columns])

        categorical_columns = process_data.select_dtypes(include='object').columns
        process_data = pd.get_dummies(process_data, columns=categorical_columns, drop_first=False) #categorical encoder
        #drop_first=True unless linear models like Logistic regression

        if inplace:
            self.data = process_data
        else:
            return process_data