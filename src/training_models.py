import pandas as pd

from src.data_prep import DiabetesPreprocessor
from src.models import (
    train_elasticnet,
    train_gradient_boosting,
    train_lasso,
    train_linear_regression,
    train_random_forest,
    train_random_forest_randomized,
    train_ridge,
    train_xgboost,
    train_xgboost_bayesian,
)


def train_all_models(df_path="data/raw/diabetes.csv"):
    df = pd.read_csv(df_path)
    pre = DiabetesPreprocessor(df)

    X_train, X_test, y_train, y_test, outliers = pre.process_with_outliers()

    (
        X_train_no,
        X_test_no,
        y_train_no,
        y_test_no,
        outliers_no,
    ) = pre.process_without_outliers()

    model_funcs = [
        train_elasticnet,
        train_linear_regression,
        train_gradient_boosting,
        train_lasso,
        train_random_forest,
        train_random_forest_randomized,
        train_ridge,
        train_xgboost,
        train_xgboost_bayesian,
    ]

    results = []

    for func in model_funcs:
        results.append(func(X_train, X_test, y_train, y_test))

        results.append(
            func(
                X_train_no,
                X_test_no,
                y_train_no,
                y_test_no,
                suffix="_iqr",
            )
        )

    return pd.DataFrame(results)


print(train_all_models())
