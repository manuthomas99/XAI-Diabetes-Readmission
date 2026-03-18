# models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
def get_models():
    models = {
    "logistic_regression": LogisticRegression(
        solver='lbfgs',          
        penalty='l2',
        C=0.01,                 
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),
    "xgboost": XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,       
        learning_rate=0.2,      
        max_depth=6,
        subsample=1.0,          
        colsample_bytree=0.6,   
        gamma=0.1,              
        reg_alpha=0.1,          
        scale_pos_weight=10.31,
        reg_lambda=1.0,           
        min_child_weight=1,
        random_state=42,
        n_jobs=-1
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,       
        max_depth=30,           
        min_samples_split=2,
        min_samples_leaf=4,     
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    "neural_network": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,            
        learning_rate='constant', 
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
}
    return models

def get_param_grids():

    param_grids = {
            "logistic_regression": {
                "model__C": [0.001, 0.01, 0.1, 1]
            },

            "random_forest": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [ 8 ,10, 20, None],
                "model__min_samples_leaf": [2 ,4, 8]
            },

            "xgboost": {
                "model__n_estimators": [ 300, 400, 500],
                "model__learning_rate": [0.2, 0.1, 0.05], 
                "model__max_depth": [5, 6, 7, 10],
                "model__subsample": [1.0],
                "model__colsample_bytree": [0.6, 0.8],
                "model__reg_lambda": [1, 5],
                "model__min_child_weight": [3, 5, 10]
            },

            "neural_network": {
                "model__hidden_layer_sizes": [(128, 64, 32), (128, 64)],
                "model__alpha": [0.001, 0.01],
                "model__learning_rate": ["adaptive","constant"]
            }
        }
    
    return param_grids