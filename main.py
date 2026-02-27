#Driver script for experiments.
import argparse
from utils.data_loader import DataLoader
DATA_PATH = "/Users/manu/Desktop/Projects/XAI-Diabetes-Readmission/data/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv"
TARGET_COLUMN_NAME = "readmitted"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with different configurations.")
    # parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    
    args = parser.parse_args()
    
    # Initialize DataLoader with the provided data path and test size
    data_loader = DataLoader(data_path=DATA_PATH, target_column_name=TARGET_COLUMN_NAME, test_size=args.test_size)
    data_loader.preprocess_data()
    # data_loader.generate_plots()