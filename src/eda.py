import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DiabetesEDA:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def overview(self):
        print("=== SHAPE ===")
        print(self.df.shape, "\n")

        print("=== INFO ===")
        print(self.df.info(), "\n")

        print("=== DESCRIBE ===")
        print(self.df.describe().T, "\n")

    def missing_values(self):
        print("=== MISSING VALUES ===")
        print(self.df.isnull().sum(), "\n")

    def correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    def feature_distributions(self):
        self.df.hist(figsize=(12, 10), bins=20)
        plt.suptitle("Feature Distributions")
        plt.show()

    def show_outliers(self):
        for col in self.df.select_dtypes(include=np.number).columns:
            plt.figure()
            self.df.boxplot(column=col)
            plt.title(f"Boxplot of {col}")
            plt.show()

    def outliers(self):
        outlier_counts = {}
        for col in self.df.select_dtypes(include=np.number).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))
            outlier_counts[col] = mask.sum()

        print("=== OUTLIERS PER FEATURE ===")
        print(pd.Series(outlier_counts))
        return outlier_counts

    def pairplot(self):
        sns.pairplot(self.df, corner=True)
        plt.show()

    def target_distribution(self):
        plt.figure(figsize=(8, 5))
        sns.histplot(data=self.df, x="Y", kde=True)
        plt.title("Target Variable (Y) Distribution")
        plt.show()


def run_eda(dataset_path: str):
    df = pd.read_csv(dataset_path)
    eda = DiabetesEDA(df)

    eda.overview()
    eda.missing_values()
    eda.correlation_heatmap()
    eda.feature_distributions()
    eda.show_outliers()
    eda.outliers()
    eda.pairplot()
    eda.target_distribution()


run_eda("data/raw/diabetes.csv")
