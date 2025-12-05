import numpy as np
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBRegressor


def eval_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def train_linear_regression(X_train, X_test, y_train, y_test, suffix=""):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    dump(model, f"models/linear_regression{suffix}.joblib")
    return eval_model(f"Linear Regression{suffix}", y_test, preds)


def train_random_forest(X_train, X_test, y_train, y_test, suffix=""):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
    }

    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    dump(best_model, f"models/random_forest{suffix}.joblib")
    return eval_model(f"Random Forest{suffix}", y_test, preds)


def train_gradient_boosting(X_train, X_test, y_train, y_test, suffix=""):
    param_grid_gb = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
    }

    gb = GradientBoostingRegressor(random_state=42)
    grid = GridSearchCV(gb, param_grid_gb, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    dump(best_model, f"models/gradient_boosting{suffix}.joblib")
    return eval_model(f"Gradient Boosting{suffix}", y_test, preds)


def train_ridge(X_train, X_test, y_train, y_test, suffix=""):
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    dump(model, f"models/ridge{suffix}.joblib")
    return eval_model(f"Ridge {suffix}", y_test, preds)


def train_lasso(X_train, X_test, y_train, y_test, suffix=""):
    model = Lasso(alpha=0.01, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    dump(model, f"models/lasso{suffix}.joblib")
    return eval_model(f"Lasso {suffix}", y_test, preds)


def train_elasticnet(X_train, X_test, y_train, y_test, suffix=""):
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    dump(model, f"models/elasticnet{suffix}.joblib")
    return eval_model(f"ElasticNet {suffix}", y_test, preds)


def train_xgboost(X_train, X_test, y_train, y_test, suffix=""):
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    dump(model, f"models/xgboost{suffix}.joblib")
    return eval_model(f"XGBoost{suffix}", y_test, preds)


def train_random_forest_randomized(X_train, X_test, y_train, y_test, suffix=""):
    param_dist_rf = {
        "n_estimators": [200, 300, 500, 700, 1000],
        "max_depth": [3, 5, 7, 9, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    rf = RandomForestRegressor(random_state=42)

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist_rf,
        n_iter=30,
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    search.fit(X_train, y_train)
    best = search.best_estimator_
    preds = best.predict(X_test)

    dump(best, f"models/random_forest_randomized{suffix}.joblib")
    return eval_model(f"Random Forest (RandomizedSearch){suffix}", y_test, preds)


def train_xgboost_bayesian(X_train, X_test, y_train, y_test, suffix=""):
    base = XGBRegressor(objective="reg:squarederror", random_state=42)

    search_spaces = {
        "n_estimators": Integer(200, 800),
        "max_depth": Integer(2, 8),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "subsample": Real(0.5, 1.0),
        "colsample_bytree": Real(0.5, 1.0),
    }

    bayes = BayesSearchCV(
        estimator=base,
        search_spaces=search_spaces,
        n_iter=30,
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    bayes.fit(X_train, y_train)
    best = bayes.best_estimator_
    preds = best.predict(X_test)

    dump(best, f"models/xgboost_bayesian{suffix}.joblib")
    return eval_model(f"XGBoost (Bayesian){suffix}", y_test, preds)
