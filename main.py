import argparse
import random
import joblib
import os
import warnings
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from utils.explainability import explain_shap, explain_shap_single
from utils.data_loader import DataLoader
from utils.models import get_models, get_param_grids

os.makedirs("outputs", exist_ok=True)
warnings.filterwarnings('ignore')



DATA_PATH = "./data/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv"
TARGET_COLUMN_NAME = "readmitted"
NUMERICAL_COLS = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                  'num_medications', 'number_outpatient', 'number_emergency',
                  'number_inpatient', 'number_diagnoses']  #Numerical columns after preprocessing is done
TEST_SIZE = 0.15
RANDOM_STATE = 42

def main(args):
    
    # 1. Load & Preprocess 
    print("Loading and preprocessing data...")
    data_loader = DataLoader(data_path=DATA_PATH, target_column_name=TARGET_COLUMN_NAME, test_size=TEST_SIZE)
    data_loader.preprocess_data()

    # 2. Prepare X, y
    data = data_loader.data
    X = data.drop(columns=[TARGET_COLUMN_NAME])
    y = data[TARGET_COLUMN_NAME]
    print(f"Data shape: {X.shape}, Target distribution:\n{y.value_counts()}\n")
    
    # Print data statistics:
    print("Data statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Numerical features: {len(NUMERICAL_COLS)}")
    print(f"  Categorical features: {X.shape[1] - len(NUMERICAL_COLS)}")
    print(f"  Class distribution:\n{y.value_counts(normalize=True)}\n")

    #get number of readmission vs non-readmission cases for XGBoost's scale_pos_weight 
    num_readmitted, num_not_readmitted = y.value_counts().get(1, 0), y.value_counts().get(0, 0)
    print(f"  Ratio of non-readmission (0) cases to readmission (1) cases: {num_not_readmitted / num_readmitted:.2f}\n")
    print(f"  %  of non-readmission (0) cases to readmission (1) cases: {y.value_counts(normalize=True).get(0, 0) * 100:.2f}% vs {y.value_counts(normalize=True).get(1, 0) * 100:.2f}%\n")
    
    # 3. Train/Test Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 5. Train Models 
    models = get_models()
    models = {name: model for name, model in models.items() if name in args.models} 

    param_grids = get_param_grids()
    best_pipelines = {}
    results = {}

    for name, model in models.items():
        print(f"{'='*50}")
        print(f"Training: {name}")

        if args.balance:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        
        #Fit and optionally tune
        if args.tune:
            print(f"Running hyperparameter tuning (n_iter={args.n_iter})...")
            
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grids[name],
                n_iter=args.n_iter,
                cv=5,
                scoring= 'f1_macro', #or 'average_precision'
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=1
            )

            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_

            print(f"  Best params: {search.best_params_}")
            print(f"  Best CV F1:  {search.best_score_:.4f}")
        
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)

        # 6. Evaluate 
        y_pred = best_pipeline.predict(X_test) #SMOTE automatically skipped 
        y_proba = best_pipeline.predict_proba(X_test)[:, 1] if hasattr(best_pipeline.named_steps['model'], "predict_proba") else None

        print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            print(f"  ROC-AUC: {auc:.4f}")
        else:
            auc = None

        best_pipelines[name] = best_pipeline
        results[name] = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "roc_auc": auc
        }

        
    # 7. Summary 
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for name, result in results.items():
        f1 = result['classification_report']['macro avg']['f1-score']
        auc = result['roc_auc']
        print(f"{name:25s} | F1 (macro): {f1:.4f} | ROC-AUC: {f'{auc:.4f}' if auc is not None else 'N/A'}")

    # 8. Save Models (optional) 
    if args.save_models:
        print("\nSaving models...")
        for name, model in best_pipelines.items():
            path = f"Outputs/{name}_best.pkl"
            joblib.dump(model, path)
            print(f"  Saved: {path}")
        joblib.dump(pipeline, "Outputs/pipeline.pkl")
        print("  Saved: Outputs/pipeline.pkl")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with different configurations.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning.")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of iterations for RandomizedSearchCV.")
    parser.add_argument("--save_models", action="store_true", help="Save best models to disk.")
    parser.add_argument("--balance", action="store_true", help="Include SMOTE in the pipeline for class balancing.")
    parser.add_argument("--models", nargs="+", choices=["logistic_regression", "random_forest", "xgboost", "neural_network","balanced_rf"], default=["logistic_regression", "random_forest", "xgboost", "neural_network","balanced_rf"], help="Specify which models to run.")
    parser.add_argument("--explain", action="store_true", help="Run explainability analyses (SHAP, LIME).")
    args = parser.parse_args()

    random.seed(RANDOM_STATE)

    main(args)