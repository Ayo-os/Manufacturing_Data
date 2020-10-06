# -*- coding: utf-8 -*-
"""
https://archive.ics.uci.edu/ml/datasets/SECOM

Created on Fri May 13 21:27:04 2020

@author: AYO

"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

class Analyzer:
    
    def __init(self):
        pass
    
    def get_data(self):
        """
        Read in data files.
        Return data in numpy array for all numerical value and
        data with 'other values' return in DataFrame object
        """
        fname = 'C:/Users/OSENIA/Downloads/PDF Embed/secom.data'
        
        self.data = np.loadtxt(fname)
            
        flabels = open('C:/Users/OSENIA/Downloads/PDF Embed/secom_labels.data')
        self.labels_df = pd.read_table(flabels, engine = 'python', sep = '\s') 
        flabels.close()
    
    def split_data(self):
        """
        Convert numpy data array into DataFrame
        """
        data = pd.DataFrame(data = self.data)
        data = data.drop(index = len(data) - 1)
        """
        Get DataFrame info as str data
        """
        buffer = io.StringIO()
        data.info(buf = buffer)
        self.data_info = buffer.getvalue()
        
        """
        split data into train and test datasets. Can also use:
        sklearn.datasets function train_test_splits()
        maybe only on numpy arrays tho. check
        
        """
        thirty_per = (0.3 * len(data))
        mid_min = ((len(data)/2) - (0.5*thirty_per))
        mid_plus = ((len(data)/2) + (0.5*thirty_per))
        data_test = data.iloc[int(mid_min) : int(mid_plus)]
    
        data_new = data.merge(data_test, how ='left', indicator=True)
        data_train = data_new[data_new['_merge'] == 'left_only']
        data_train = data_train.drop(columns = '_merge')
        
        self.data_train = data_train
        self.data_test = data_test
    
    def split_labels(self):
        """
        split labels into train and test
        """
        data = self.labels_df
        
        thirty_per = (0.3 * len(data))
        mid_min = ((len(data)/2) - (0.5*thirty_per))
        mid_plus = ((len(data)/2) + (0.5*thirty_per))
        data_test = data.iloc[int(mid_min) : int(mid_plus)]
    
        data_new = data.merge(data_test, how ='left', indicator=True)
        data_train = data_new[data_new['_merge'] == 'left_only']
        data_train = data_train.drop(columns = '_merge')
    
        self.labels_train = data_train
        self.labels_test = data_test.iloc[:,0]
    
    def data_expl(self):
        """
        Find columns missing data. Sort by % missing
        """
        data = self.data_train
        
        data_miss = data.columns[data.isnull().any()]
        miss_count = data.isnull().sum()/len(data)
        miss_count = miss_count[miss_count > 0]
        miss_count.sort_values(inplace=True)
        
        """
        Convert miss=Series to DatFrame. Instead of index count,
        Left column is column number
        """
        miss_count = miss_count.to_frame()
        miss_count.columns = ['% missing']
        miss_count.index.names = ['column #']
        miss_count['column #'] = miss_count.index

        self.miss_count = miss_count
        
    def data_expl2(self):
        """
        Combine Data and labels
        calculate correlations to labels
        """
        data = self.data_train.copy()
        data1 = self.labels_train
        
        data[len(data.columns)+1] = data1.iloc[:,0][0:len(data)]
        
        #Drop n/a correlations
        corr = data.corr()
        corr = pd.DataFrame(corr)
        corr = corr.dropna(how = 'all')
        corr = corr.dropna(axis = 1, how = 'all')
        
        corr_miss = corr.isnull().sum()/len(corr)
        corr_miss = corr_miss[corr_miss > 0]
        corr_miss.sort_values(inplace=True)
        
        self.corr = corr
        self.corr_miss = corr_miss
        
        #Highest and lowest correlated parameters
        self.label_high = corr.iloc[:,-1].sort_values(ascending=False)[:30]
        self.label_low = corr.iloc[:,-1].sort_values(ascending=False)[-10:]    
    
    def quick_proc(self):
        """
        Replace nans with mean value in column
        """
        data = self.data_train
        data_test = self.data_test
        
        data = data.fillna(data.mean())
        
        data_test = data_test.fillna(data.mean())
        
        self.data_train1 = data
        self.data_test1 = data_test
        
    def predict_LR(self):
        """
        predict results using Logistic Regression
        """
        data = self.data_train1
        labels = self.labels_train
        data_test = self.data_test1
        labels_test = self.labels_test
        
        model = LogisticRegression()
        model.fit(data, labels.iloc[:,0])
        
        prediction = model.predict(data_test)
        model_score = model.score(data_test, labels_test)
        
        self.LR_prediction = prediction
        self.LR_score = model_score
    
    def predict_SVM(self):
        """
        predict results using Support Vector Machines
        """
        data = self.data_train1
        labels = self.labels_train
        data_test = self.data_test1
        labels_test = self.labels_test
        
        model = svm.LinearSVC()
        model.fit(data, labels.iloc[:,0])
        prediction = model.predict(data_test)   
        model_score = model.score(data_test, labels_test)
        
        self.SVM_prediction = prediction
        self.SVM_score = model_score
    
    def predict_RF(self):
        """
        predict results using Random Forest
        """
        data = self.data_train1
        labels = self.labels_train
        data_test = self.data_test1
        labels_test = self.labels_test
        
        model = RandomForestClassifier()
        model.fit(data, labels.iloc[:,0])
        prediction = model.predict(data_test)   
        model_score = model.score(data_test, labels_test)
        
        self.RF_prediction = prediction
        self.RF_score = model_score
    
    def predict_NN(self):
        """
        predict results using Neural Network
        """
        data = self.data_train1
        labels = self.labels_train
        data_test = self.data_test1
        labels_test = self.labels_test
        
        model = MLPClassifier()
        model.fit(data, labels.iloc[:,0])
        prediction = model.predict(data_test)   
        model_score = model.score(data_test, labels_test)
        
        self.NN_prediction = prediction
        self.NN_score = model_score
        
    def predict_XGB(self):
        """
        predict results using XGBoost
        """
        pass

if __name__ == '__main__':    
    F1 = Analyzer()
    F1.get_data()
    F1.split_data()
    F1.split_labels()
    F1.data_expl()
    F1.data_expl2()
    F1.quick_proc()
    F1.predict_LR()
    #F1.predict_SVM()
    #F1.predict_RF()
    #F1.predict_NN()
    #F1.predict_XBG()
    
        


















