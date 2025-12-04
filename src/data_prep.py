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

    # OUTLIER DETECTION (IQR)
    def detect_outliers_iqr(self):
        numeric = self.df.select_dtypes(include=np.number)
        Q1 = numeric.quantile(0.25)
        Q3 = numeric.quantile(0.75)
        IQR = Q3 - Q1

        mask = (numeric < (Q1 - 1.5 * IQR)) | (numeric > (Q3 + 1.5 * IQR))
        count = mask.sum()

        self.outliers_iqr = count[count > 0]
        print("\nDetected outliers using IQR:")
        print(self.outliers_iqr)
        return self.outliers_iqr

    # TRAIN–TEST SPLIT
    def split(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    # SCALING USING ROBUSTSCALER
    def scale(self, X_train, X_test):
        numeric_cols = X_train.select_dtypes(include=np.number).columns

        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])

        X_test_scaled = X_test.copy()
        X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        return X_train_scaled, X_test_scaled

    # PIPELINE
    def process(self):
        self.detect_outliers_iqr()
        X_train, X_test, y_train, y_test = self.split()
        X_train_scaled, X_test_scaled = self.scale(X_train, X_test)
        self.save_scaler()

        return X_train_scaled, X_test_scaled, y_train, y_test, self.outliers_iqr

    def save_scaler(self, path="models/scaler.joblib"):
        dump(self.scaler, path)


df = pd.read_csv("data/raw/diabetes.csv")

pre = DiabetesPreprocessor(df)
X_train, X_test, y_train, y_test, outliers = pre.process()
