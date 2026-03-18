import argparse
import random
import joblib
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline, clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from utils.explainability import explain_shap, explain_shap_single
from utils.data_loader import DataLoader
from utils.models import get_models, get_param_grids

os.makedirs("Outputs", exist_ok=True)
warnings.filterwarnings('ignore')



DATA_PATH = "data/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv"
TARGET_COLUMN_NAME = "readmitted"
NUMERICAL_COLS = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                  'num_medications', 'number_outpatient', 'number_emergency',
                  'number_inpatient', 'number_diagnoses']  #Numerical columns after preprocessing is done
TEST_SIZE = 0.15
RANDOM_STATE = 42

def check_overfitting(best_pipeline, X_train, y_train, X_test, y_test, model_name):
    """Compare train vs test metrics to detect overfitting."""

    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred  = best_pipeline.predict(X_test)

    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    test_report  = classification_report(y_test,  y_test_pred,  output_dict=True)

    train_f1 = train_report['macro avg']['f1-score']
    test_f1  = test_report['macro avg']['f1-score']

    model = best_pipeline.named_steps['model']
    if hasattr(model, "predict_proba"):
        train_auc = roc_auc_score(y_train, best_pipeline.predict_proba(X_train)[:, 1])
        test_auc  = roc_auc_score(y_test,  best_pipeline.predict_proba(X_test)[:, 1])
    else:
        train_auc = test_auc = None

    F1_GAP_THRESHOLD  = 0.05   
    AUC_GAP_THRESHOLD = 0.05

    f1_gap  = train_f1 - test_f1
    auc_gap = (train_auc - test_auc) if train_auc is not None else None

    print(f"\n  {'='*55}")
    print(f"  Overfitting Check — {model_name}")
    print(f"  {'='*55}")
    print(f"  {'Metric':<12} {'Train':>10} {'Test':>10} {'Gap':>10}")
    print(f"  {'-'*55}")
    print(f"  {'F1 (macro)':<12} {train_f1:>10.4f} {test_f1:>10.4f} {f1_gap:>+10.4f}", 
          "⚠ OVERFIT" if f1_gap > F1_GAP_THRESHOLD else "✓ OK")
    
    if auc_gap is not None:
        print(f"  {'ROC-AUC':<12} {train_auc:>10.4f} {test_auc:>10.4f} {auc_gap:>+10.4f}",
              "⚠ OVERFIT" if auc_gap > AUC_GAP_THRESHOLD else "✓ OK")
    print(f"  {'='*55}")

    return {"train_f1": train_f1, "test_f1": test_f1, "f1_gap": f1_gap,
            "train_auc": train_auc, "test_auc": test_auc}

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

    
    # 3. Train/Test Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    #get number of readmission vs non-readmission cases for XGBoost's scale_pos_weight using train data
    num_readmitted, num_not_readmitted = y_train.value_counts().get(1, 0), y_train.value_counts().get(0, 0)
    print(f"  Ratio of non-readmission (0) cases to readmission (1) cases: {num_not_readmitted / num_readmitted:.2f}\n")
    print(f"  %  of non-readmission (0) cases to readmission (1) cases: {y_train.value_counts(normalize=True).get(0, 0) * 100:.2f}% vs {y_train.value_counts(normalize=True).get(1, 0) * 100:.2f}%\n")
    


    if args.predict_only:
        print("Predict-only mode: loading saved models...")
        for name in args.models:
            path = f"Outputs/{name}_best.pkl"
            if not os.path.exists(path):
                print(f"  [SKIP] No saved model found at {path}")
                continue

            print(f"\n{'='*50}")
            print(f"Evaluating: {name}")
            best_pipeline = joblib.load(path)
            print(f"  Loaded: {path}")

            # Predict
            y_pred  = best_pipeline.predict(X_test)
            y_proba = best_pipeline.predict_proba(X_test)[:, 1] \
                      if hasattr(best_pipeline.named_steps['model'], "predict_proba") else None

            print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
            print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

            if y_proba is not None:
                auc = roc_auc_score(y_test, y_proba)
                print(f"  ROC-AUC: {auc:.4f}")

            # SHAP
            if args.explain:
                positive_indices = np.where(y_pred == 1)[0]

                if len(positive_indices) == 0:
                    print("  No positive predictions found — skipping individual explanation.")
                else:
                    if y_proba is not None:
                        positive_indices = sorted(
                            positive_indices,
                            key=lambda i: y_proba[i],
                            reverse=True        
                        )

                    feature_names = X_train.columns.tolist()

                    print(f"  Running SHAP for {name}...")
                    shap_values, shap_explainer = explain_shap(
                        pipeline=best_pipeline,
                        X_train=X_train,
                        X_test=X_test,
                        feature_names=feature_names,
                        model_name=name
                    )

                    X_test_sc = best_pipeline.named_steps['scaler'].transform(X_test)
                    
                    for rank, idx in enumerate(positive_indices[:3]):
                        true_label  = y_test.iloc[idx]
                        pred_proba  = f"{y_proba[idx]:.4f}" if y_proba is not None else "N/A"

                        print(f"\n  Explaining predicted positive #{rank+1}")
                        print(f"    Test index   : {idx}")
                        print(f"    True label   : {true_label}  ({'TP' if true_label == 1 else 'FP'})")
                        print(f"    Pred proba   : {pred_proba}")

                        explain_shap_single(
                            shap_explainer,
                            shap_values,
                            X_test_sc,
                            feature_names,
                            index=idx,
                            model_name=name
                        )
                
        check_overfitting(best_pipeline, X_train, y_train, X_test, y_test, model_name=name)
        return          
    
    # 5. Train Models 
    models = get_models()
    models = {name: model for name, model in models.items() if name in args.models} 

    param_grids = get_param_grids()
    best_pipelines = {}
    results = {}

    for name, model in models.items():
        print(f"{'='*50}")
        print(f"Training: {name}")


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
                scoring= 'balanced_accuracy', #, #or f1_macro
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=1,
                return_train_score=True
            )

            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            cv_results = pd.DataFrame(search.cv_results_)
            cv_results["overfit_gap"] = cv_results["mean_train_score"] - cv_results["mean_test_score"]

            cv_results["val_score_norm"] = (
                (cv_results["mean_test_score"] - cv_results["mean_test_score"].min()) /
                (cv_results["mean_test_score"].max() - cv_results["mean_test_score"].min())
            )
            cv_results["gap_norm"] = (
                (cv_results["overfit_gap"] - cv_results["overfit_gap"].min()) /
                (cv_results["overfit_gap"].max() - cv_results["overfit_gap"].min())
            )

            cols = ["params", "mean_train_score", "mean_test_score", "overfit_gap"]

            top = (cv_results[cols]
                [cv_results["overfit_gap"] <= 0.1]   # only generalising models
                .sort_values("mean_test_score", ascending=False)
                .round(4)
                .head(15))
            filtered = cv_results[cv_results["overfit_gap"] <= 0.1]
            if filtered.empty:
                print("No models passed the overfitting threshold. Selecting best overall model.")
                filtered = cv_results
            best_idx = filtered.sort_values("mean_test_score", ascending=False).index[0]
            best_params = search.cv_results_["params"][best_idx]
            best_pipeline = clone(search.estimator)
            best_pipeline.set_params(**best_params)
            best_pipeline.fit(X_train, y_train)
            print(top.to_string(index=False))

            best_row = top.iloc[0]
            print("\nBest params:", best_row["params"])
            print("Val F1 (macro):", best_row["mean_test_score"])
            print("Overfit gap:   ", best_row["overfit_gap"])
        
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)

        # 6. Evaluate 
        y_pred = best_pipeline.predict(X_test) #SMOTE automatically skipped 
        y_proba = best_pipeline.predict_proba(X_test)[:, 1] if hasattr(best_pipeline.named_steps['model'], "predict_proba") else None

        print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        overfitting_stats = check_overfitting(
                best_pipeline, X_train, y_train, X_test, y_test, model_name=name
        )
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            print(f"  ROC-AUC: {auc:.4f}")
        else:
            auc = None

        best_pipelines[name] = best_pipeline
        results[name] = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "roc_auc": auc,
            "overfitting": overfitting_stats
        }

        # 7. Explainability
        if args.explain:
            feature_names = X_train.columns.tolist()

            print(f"  Running SHAP for {name}...")
            shap_values, shap_explainer = explain_shap(
                pipeline=best_pipeline,
                X_train=X_train,
                X_test=X_test,
                feature_names=feature_names,
                model_name=name
            )
            
    # 8. Summary 
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
    parser.add_argument("--models", nargs="+", choices=["logistic_regression", "random_forest", "xgboost", "neural_network","balanced_rf"], default=["logistic_regression", "random_forest", "xgboost", "neural_network","balanced_rf"], help="Specify which models to run.")
    parser.add_argument("--explain", action="store_true", help="Run explainability analyses (SHAP, LIME).")
    parser.add_argument("--predict_only", action="store_true", help="Skip training and only run predictions with saved models.")
    args = parser.parse_args()

    random.seed(RANDOM_STATE)

    main(args)