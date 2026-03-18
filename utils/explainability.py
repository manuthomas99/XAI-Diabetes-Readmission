import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_model_from_pipeline(pipeline):
    """Extract the final model step from an imblearn/sklearn Pipeline."""
    return pipeline.named_steps['model']


def get_scaler_transform(pipeline, X):
    """Apply only the scaler step from the pipeline to X."""
    return pipeline.named_steps['scaler'].transform(X)


def explain_shap(pipeline, X_train, X_test, feature_names, model_name, max_display=20):
    """
    Compute and plot SHAP values.
    """
    model      = get_model_from_pipeline(pipeline)
    X_train_sc = get_scaler_transform(pipeline, X_train)
    X_test_sc  = get_scaler_transform(pipeline, X_test)

    X_train_df = pd.DataFrame(X_train_sc, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test_sc,  columns=feature_names)

    # Choose explainer based on model type
    if model_name in ('xgboost', 'random_forest'):
        explainer  = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)

        # Random forest returns list [class0, class1] — take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    elif model_name == 'logistic_regression':
        explainer   = shap.LinearExplainer(model, X_train_df)
        shap_values = explainer.shap_values(X_test_df)

    else:  
        background  = shap.sample(X_train_df, 100)
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test_df[:200])[1]  # class 1 only

    # Plotting
    # 1. Global feature importance
    plt.figure()
    shap.summary_plot(shap_values, X_test_df, max_display=max_display, show=False)
    plt.title(f"SHAP Summary — {model_name}")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_summary_{model_name}.png", dpi=150)
    plt.close()

    # 2. Bar plot — mean absolute SHAP
    plt.figure()
    shap.summary_plot(shap_values, X_test_df, plot_type='bar', max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_importance_{model_name}.png", dpi=150)
    plt.close()

    print(f"  SHAP plots saved for {model_name}")
    return shap_values, explainer


def explain_shap_single(shap_explainer, shap_values, X_scaled, feature_names, index, model_name):
    """
    Waterfall plot for a single prediction with proper label rendering.
    """
    print(f"    Generating waterfall explanation for sample index: {index}")
    
    if isinstance(shap_values, shap.Explanation):
        sv = shap_values[index]
    else:
        base_value = shap_explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
            values     = shap_values[1][index]   # class-1 SHAP values
        else:
            values     = shap_values[index]

        sv = shap.Explanation(
            values       = values,
            base_values  = base_value,
            data         = X_scaled[index],
            feature_names= feature_names
        )

    fig, ax = plt.subplots(figsize=(12, 7))   

    shap.plots.waterfall(sv, max_display=12, show=False)

    
    longest = max(len(n) for n in feature_names)
    left_margin = min(0.45, max(0.25, longest * 0.012)) 

    plt.gcf().set_size_inches(13, 7)
    plt.gcf().subplots_adjust(left=left_margin) 
    plt.tight_layout(rect=[left_margin, 0, 1, 1])

    out_path = f"Outputs/{model_name}_shap_sample{index}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")  
    plt.close()
    print(f"    Saved: {out_path}")