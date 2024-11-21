#!/usr/bin/env python
# coding: utf-8

# # All hyper parameters

# In[ ]:


# Necessary imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Data scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "RandomForest": RandomForestClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "LightGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Hyperparameters for each model
param_grids = {
    "RandomForest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    "CatBoost": {
        'iterations': [100, 200],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5]
    },
    "LightGBM": {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'min_child_samples': [20, 50, 100]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
}

# K-fold cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearchCV and GridSearchCV for each model
best_estimators = {}

for model_name in models:
    model = models[model_name]
    param_grid = param_grids[model_name]
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=kf, n_jobs=-1, verbose=1)
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_name} (RandomizedSearchCV): {random_search.best_params_}")
    
    # GridSearchCV based on best parameters from RandomizedSearchCV
    grid_search = GridSearchCV(model, param_grid=random_search.best_params_, cv=kf, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_name} (GridSearchCV): {grid_search.best_params_}")
    
    # Save the best estimator
    best_estimators[model_name] = grid_search.best_estimator_

# Evaluate on test set
for model_name, best_estimator in best_estimators.items():
    y_pred = best_estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy for {model_name}: {accuracy:.4f}")
     
     


# # Data_imbalance

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv("C:/Users/govar/Downloads/creditcard.csv")
data.head()


# In[4]:


data['Class'].nunique


# In[5]:


data.isnull().sum()


# In[6]:


x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[7]:


sns.countplot(y)


# In[15]:


pip install imblearn


# In[16]:


pip install -U scikit-learn


# In[ ]:





# In[ ]:




