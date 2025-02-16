import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor

model_dict = {
    'classification': [TabPFNClassifier, AutoTabPFNClassifier], 
    'regression': [TabPFNRegressor, AutoTabPFNRegressor]
    }

class DataLoader:
    def __init__(self, loader=None):
        self.loader = loader

    def create_data_loader(self, dataset):
        class _DataLoader:
            def __init__(self, dataset):
                self.data, self.target = dataset
        return _DataLoader(dataset)

    def __call__(self):
        if hasattr(self.loader, '__call__'):
            data = self.loader()
            return data
        elif type(self.loader) is tuple:
            data = self.create_data_loader(self.loader)
            return data
        else:
            raise ValueError("There is no data file.")

class Runner:
    def __init__(self, data_dict, auto=False):
        self.data_dict = data_dict
        self.metric = {}

        if type(self.data_dict) is dict:
            self.task = self.data_dict['task']
            self.data = self.data_dict['loader']()
            self.split_data(self.data)
        else:
            raise ValueError("There is no data in 'data_dict'.")

        if self.task in model_dict.keys():
            self.model = model_dict[self.task][auto]()
        else:
            raise ValueError("choose task: 'classification' or 'regression'.")

    def split_data(self, data):
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.5, random_state=42)   
        print(f"Data '{self.data_dict['name']}' is loaded.")     
    
    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
        print(f"Model {self.model} training is done.")

    def eval_model(self):
        # Predict labels
        predictions = self.model.predict(self.X_test)

        if self.task == 'classification':
            # Predict probabilities
            prediction_probabilities = self.model.predict_proba(self.X_test)
            if len(set(self.y_test)) > 2: # multi-label
                roc_auc = roc_auc_score(self.y_test, prediction_probabilities, multi_class='ovr')
            else: # binary
                roc_auc = roc_auc_score(self.y_test, prediction_probabilities[:, 1])
            self.metric['ROC AUC'] = roc_auc
            self.metric['Accuracy'] = accuracy_score(self.y_test, predictions)
            
        elif self.task == 'regression':
            rmse = root_mean_squared_error(self.y_test, predictions)
            mae = mean_absolute_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            self.metric['Root Mean-Squared Error'] = rmse
            self.metric['Mean-Absolute Error'] = mae
            self.metric['R2-score'] = r2
        else:
            raise NotImplementedError
        
        print("=" * 30)
        for metric_name in self.metric.keys():
            print(f"{metric_name}: {self.metric[metric_name]}")
        print("=" * 30)

if __name__ == '__main__':

    data_list= [
        {'name': 'Breast cancer', 'loader': DataLoader(load_breast_cancer), 'task': 'classification'},
        {'name': 'Diabetes', 'loader': DataLoader(load_diabetes), 'task': 'regression'},
        {'name': 'Wine', 'loader': DataLoader(load_wine), 'task': 'classification'}
        ]

    for data_dict in data_list:
        runner = Runner(data_dict, auto=False)
        runner.fit_model()
        runner.eval_model()