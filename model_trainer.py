from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from joblib import dump
import xgboost as xgb
from scipy.stats import randint
import lightgbm as lgb

def train_and_evaluate_randf_simple(X_train, X_test, y_train, y_test, model_path):
    """Trains a RandomForestClassifier and evaluates its accuracy."""
    print("Fitting model ...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.4f}')
    
    # Save the trained model
    dump(model, model_path)


def train_and_evaluate_randf_optimised(X_train, X_test, y_train, y_test, model_path):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Optimized Model Accuracy: {accuracy:.4f}')

    # Save the best model
    dump(best_model, model_path)

def train_and_evaluate_randf_gpu_optimized(X_train, X_test, y_train, y_test, model_path):
    # Prepare the parameter grid for RandomizedSearch
    param_grid = {
        'num_leaves': randint(31, 150),
        'max_depth': randint(3, 10),
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.5, 0.7, 0.9],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'device': ['gpu'],  # Indicate to use GPU
        'gpu_platform_id': [0],
        'gpu_device_id': [0]
    }

    # Initialize LightGBM model
    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', 
                                   random_state=42, metric='binary_logloss')

    # Randomized search for hyperparameter tuning
    randomized_search = RandomizedSearchCV(lgb_model, param_distributions=param_grid, 
                                           n_iter=10, cv=3, scoring='accuracy', 
                                           n_jobs=-1, verbose=1)
    randomized_search.fit(X_train, y_train)

    # Best model after RandomizedSearch
    best_model = randomized_search.best_estimator_

    # Making predictions
    y_pred = best_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Optimized Model Accuracy with LightGBM: {accuracy:.4f}')

    # Save the best model
    dump(best_model, model_path)