import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBRegressor

from data_prep import DiabetesPreprocessor

df = pd.read_csv("data/raw/diabetes.csv")
pre = DiabetesPreprocessor(df)
X_train, X_test, y_train, y_test, outliers = pre.process()


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return [name, mae, mse, rmse, r2]


results = []

# Train Models
lr = LinearRegression().fit(X_train, y_train)
results.append(evaluate("Linear Regression", y_test, lr.predict(X_test)))

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
results.append(evaluate("Ridge", y_test, ridge.predict(X_test)))

lasso = Lasso(alpha=0.01).fit(X_train, y_train)
results.append(evaluate("Lasso", y_test, lasso.predict(X_test)))

elastic = ElasticNet(alpha=0.01).fit(X_train, y_train)
results.append(evaluate("ElasticNet", y_test, elastic.predict(X_test)))

rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
results.append(evaluate("Random Forest", y_test, rf.predict(X_test)))

gb = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
results.append(evaluate("Gradient Boosting", y_test, gb.predict(X_test)))

# XGBoost with BayesSearchCV
xgb_base = XGBRegressor(objective="reg:squarederror", random_state=42)

search_spaces_xgb = {
    "n_estimators": Integer(200, 800),
    "max_depth": Integer(2, 8),
    "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "subsample": Real(0.5, 1.0),
    "colsample_bytree": Real(0.5, 1.0),
}

bayes_search_xgb = BayesSearchCV(
    estimator=xgb_base,
    search_spaces=search_spaces_xgb,
    n_iter=30,
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

bayes_search_xgb.fit(X_train, y_train)
print("Best XGB params (Bayesian):", bayes_search_xgb.best_params_)

xgb_best_bayes = bayes_search_xgb.best_estimator_
pred_xgb_bayes = xgb_best_bayes.predict(X_test)

results.append(evaluate("XGB Bayesian", y_test, pred_xgb_bayes))

results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2"])
print("\n===== MODEL RESULTS =====")
print(results_df)
