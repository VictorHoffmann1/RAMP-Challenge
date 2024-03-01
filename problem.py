import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

problem_title = "Predicting the property value of French buildings\
      - regression"
_target_column_name = 'Valeur fonciere'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.Estimator()


score_types = [
    rw.score_types.RMSE(),
    rw.score_types.RelativeRMSE(name='rel_rmse'),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
    return cv.split(X)


def _get_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), sep=",")
    data = data.dropna(subset=[_target_column_name])
    X = data.drop([_target_column_name], axis=1)
    y_array = data[_target_column_name].str.replace(',', '.').astype(float).values
    return X, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _get_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _get_data(path, f_name)
