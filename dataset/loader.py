from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_digits,
    load_iris,
    make_classification,
    fetch_openml
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np

class LoadData:
    def __init__(self, scaler=None, dataset=''):
        self.scaler = scaler
        self.dataset = dataset
        self.data = None
        self.target = None

        if dataset:
            self.load_data()

    def load_data(self):
        dataset_loaders = {
            "breast_cancer": load_breast_cancer,
            "make_classification": make_classification,
            "ionosphere": fetch_openml,
            "wine": load_wine,
            "digits": load_digits,
            "iris": load_iris,
        }

        if self.dataset == 'make_classification':
            self.data, self.target = make_classification(
                n_samples=5000, n_features=100, n_classes=2, n_informative=10
            )

        elif self.dataset == "ionosphere":
            self.data, self.target = fetch_openml(
                "ionosphere", version=1, return_X_y=True, as_frame=False, parser='liac-arff'
            )
            le = LabelEncoder()
            self.target = le.fit_transform(self.target)

        elif self.dataset == "adult":
            df = fetch_openml("adult", version=2, as_frame=True).frame

            df = df.replace("?", np.nan).dropna()

            self.target = (df["class"] == ">50K").astype(int).values

            df = df.drop(columns=["class"])

            cat_cols = df.select_dtypes(include="category").columns.tolist()
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

            if self.scaler:
                df[num_cols] = self.scaler.fit_transform(df[num_cols])

            df = df.astype(float)  # 🔥 critical: force float dtype
            self.data = df.values

        elif self.dataset in dataset_loaders:
            self.data, self.target = dataset_loaders[self.dataset](return_X_y=True)

            if self.scaler:
                self.data = self.scaler.fit_transform(self.data)

        else:
            raise ValueError(f"Dataset '{self.dataset}' is not supported.")

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target
