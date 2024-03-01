from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd


class Classifier(BaseEstimator):
    def __init__(self):

        self.transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        self.model = LinearRegression()
        self.pipe = make_pipeline(self.transformer, self.model)

    def fit(self, X, y):
        # Get numerical data
        # it's your job to figure out what to do with categorical features
        X_numerical = pd.DataFrame(X).select_dtypes(include=['number'])
        self.pipe.fit(X_numerical, y)

    def predict(self, X):
        X_numerical =  pd.DataFrame(X).select_dtypes(include=['number'])
        return self.pipe.predict(X_numerical)
