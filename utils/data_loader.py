import copy
import pandas as pd
from utils.data_visualizations import visualize_unique_counts
class DataLoader:
    def __init__(self, data_path,target_column_name, test_size=0.2):
        self.data = None
        self.target_column_name = target_column_name
        self.data_path = data_path

        self.load_data()

    def load_data(self):
        # Loading entire data from csv
        with open(self.data_path, "r") as file:
            self.data = pd.read_csv(file,na_values=['?','None'])
        return self.data
    
    
    def generate_plots(self, data=None):
        if data is None:
            data = self.data
        columns_to_plot = ["age", "race","gender"]
        for col in columns_to_plot:
            visualize_unique_counts(data, col, save_path=f"/Users/manu/Desktop/Projects/XAI-Diabetes-Readmission/Outputs/{col}_distribution.png")

    def preprocess_data(self, inplace=True):
        """
        Preprocess the data by performing the following steps:
        1. Drop irrelevant columns.
        2. Handle missing values by dropping rows with missing values in critical columns.
        3. Convert categorical variables to numerical using one-hot encoding.
        4. Scale numerical features using StandardScaler.
        """
        
        process_data = copy.deepcopy(self.data)  

        process_data.drop_duplicates(subset='patient_nbr', keep='first', inplace=True) # consider feature extraction

        # cols_to_check = ['chlorpropamide', 'acetohexamide', 'tolbutamide', 'acarbose', 'miglitol', 
        # 'troglitazone', 'tolazamide', 'glyburide-metformin', 'glipizide-metformin', 'metformin-rosiglitazone', 'metformin-pioglitazone']


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

        self.generate_plots(process_data) #generate plots before categorical encoding adds a lot of new columns

        categorical_columns = process_data.select_dtypes(include='object').columns
        process_data = pd.get_dummies(process_data, columns=categorical_columns, drop_first=False) #categorical encoder

        if inplace:
            self.data = process_data
        else:
            return process_data