import shap
import pandas as pd
import matplotlib.pyplot as plt


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


def explain_shap_single(explainer, shap_values, X_test, feature_names, index=0, model_name='model'):
    """Waterfall plot for a single prediction."""
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[index],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                        else explainer.expected_value[1],
            data=X_test_df.iloc[index],
            feature_names=feature_names
        )
    )
    plt.title(f"SHAP Waterfall — {model_name} — sample {index}")
    plt.tight_layout()
    plt.savefig(f"outputs/shap_waterfall_{model_name}_sample{index}.png", dpi=150)
    plt.close()

