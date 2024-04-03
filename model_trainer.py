from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from joblib import dump
from scipy.stats import randint, uniform
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier

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

def train_and_evaluate_multi_label_gpu_optimized(X_train, X_test, y_train, y_test, model_path):
    # Prepare the parameter grid for RandomizedSearch
    param_grid = {
        'estimator__num_leaves': randint(31, 150),
        'estimator__max_depth': randint(3, 10),
        'estimator__learning_rate': [0.01, 0.05, 0.1],
        'estimator__subsample': [0.5, 0.7, 0.9],
        'estimator__colsample_bytree': [0.5, 0.7, 0.9],
        'estimator__device': ['gpu'],  # Indicate to use GPU
        'estimator__gpu_platform_id': [0],
        'estimator__gpu_device_id': [0]
    }

    # Initialize LightGBM model
    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', 
                                   random_state=42, metric='binary_logloss')

    # Wrap the LightGBM model with MultiOutputClassifier for multi-label classification
    multi_lgb_model = MultiOutputClassifier(lgb_model, n_jobs=-1)

    # Randomized search for hyperparameter tuning
    randomized_search = RandomizedSearchCV(multi_lgb_model, param_distributions=param_grid, 
                                           n_iter=10, cv=3, scoring='accuracy', 
                                           n_jobs=-1, verbose=1)
    randomized_search.fit(X_train, y_train)

    # Best model after RandomizedSearch
    best_model = randomized_search.best_estimator_

    # Making predictions
    y_pred = best_model.predict(X_test)

    # Calculate accuracy (Note: You might want to use a different metric for multi-label problems)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Optimized Model Accuracy with LightGBM (GPU): {accuracy:.4f}')

    # Save the best model
    dump(best_model, model_path)
    
def f1_samples_scorer(y_true, y_pred):
    # Threshold predictions to convert probabilities to binary output
    y_pred_binary = (y_pred > 0.5).astype(int)
    return f1_score(y_true, y_pred_binary, average='samples')

def train_and_evaluate_multi_label_gpu_optimized_balanced(X_train, X_test, y_train, y_test, model_path):
    # Scorer for multi-label classification
    f1_scorer = make_scorer(f1_samples_scorer)

    # Expanded and more granular parameter grid
    param_grid = {
        'estimator__num_leaves': randint(31, 150),
        'estimator__max_depth': randint(3, 10),
        'estimator__learning_rate': uniform(0.01, 0.1),
        'estimator__subsample': uniform(0.5, 0.4),  # 0.5 to 0.9
        'estimator__colsample_bytree': uniform(0.5, 0.4),  # 0.5 to 0.9
        'estimator__device': ['gpu'],  # Use GPU
        'estimator__gpu_platform_id': [0],
        'estimator__gpu_device_id': [0],
        'estimator__class_weight': [None, 'balanced']  # Handling class imbalance
    }

    # Initialize LightGBM model with binary objective and early stopping
    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', 
                                   random_state=42, metric='binary_logloss', 
                                   n_estimators=10000, early_stopping_rounds=100,
                                   verbose=1)

    # Wrap the LightGBM model with MultiOutputClassifier
    multi_lgb_model = MultiOutputClassifier(lgb_model, n_jobs=-1)

    # Randomized search for hyperparameter tuning with more iterations and cross-validation folds
    randomized_search = RandomizedSearchCV(multi_lgb_model, param_distributions=param_grid, 
                                           n_iter=50, cv=5, scoring=f1_scorer, 
                                           n_jobs=-1, verbose=1, refit=True)
    randomized_search.fit(X_train, y_train)

    # Best model after RandomizedSearch
    best_model = randomized_search.best_estimator_

    # Making predictions
    y_pred = best_model.predict(X_test)

    # Calculate the F1 Score
    f1 = f1_samples_scorer(y_test, y_pred)
    print(f'Optimized Model F1 Score with LightGBM (GPU): {f1:.4f}')

    # Save the best model
    dump(best_model, model_path)