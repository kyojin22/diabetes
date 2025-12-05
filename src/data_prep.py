import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class DiabetesPreprocessor:
    def __init__(self, df: pd.DataFrame, target="Y") -> None:
        self.df = df.copy()
        self.target = target
        self.scaler = RobustScaler()
        self.outliers_iqr = None

    def detect_outliers_iqr(self):
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1

        mask = (self.df < (Q1 - 1.5 * IQR)) | (self.df > (Q3 + 1.5 * IQR))
        count = mask.sum()

        self.outliers_iqr = count[count > 0]
        return self.outliers_iqr

    def remove_outliers(self):
        Q1_iqr = self.df.quantile(0.25)
        Q3_iqr = self.df.quantile(0.75)
        IQR_iqr = Q3_iqr - Q1_iqr

        mask_iqr = ~(
            (self.df < (Q1_iqr - 1.5 * IQR_iqr)) | (self.df > (Q3_iqr + 1.5 * IQR_iqr))
        ).any(axis=1)

        return self.df[mask_iqr].reset_index(drop=True)

    def split(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def scale(self, X_train, X_test):
        numeric_cols = X_train.select_dtypes(include=np.number).columns

        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])

        X_test_scaled = X_test.copy()
        X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        return X_train_scaled, X_test_scaled

    def process_with_outliers(self):
        self.detect_outliers_iqr()
        X_train, X_test, y_train, y_test = self.split(
            X=self.df.drop(columns=[self.target]), y=self.df[self.target]
        )
        X_train_scaled, X_test_scaled = self.scale(X_train, X_test)
        self.save_scaler()

        return X_train_scaled, X_test_scaled, y_train, y_test, self.outliers_iqr

    def process_without_outliers(self):
        self.detect_outliers_iqr()
        clean_df = self.remove_outliers()
        X_train, X_test, y_train, y_test = self.split(
            X=clean_df.drop(columns=[self.target]), y=clean_df[self.target]
        )
        X_train_scaled, X_test_scaled = self.scale(X_train, X_test)
        self.save_scaler()

        return X_train_scaled, X_test_scaled, y_train, y_test, self.outliers_iqr

    def save_scaler(self, path="models/scaler.joblib"):
        dump(self.scaler, path)
