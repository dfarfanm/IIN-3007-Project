# IIN-3007 Final Project
# Predicting Breast Cancer Outcomes using Machine Learning Algorithms
## Authors: Emily Mosquera (329983) and Dominique Farfán (328313)
### Project Description: Machine Learning applied to a medical dataset. 
This project aims to predict patient survival outcomes using four machine learning algorithms: XG Boosting, Categorical Boosting, Random Forests and Gradient Boosting. We performed hyperparameter optimization with Grid Search and Randomized Search, and literary review for the implementation. We evaluated models using metrics like Recall, Precision, AUC-ROC, and F1-Score, focusing on minimizing misclassification risks in a medical (cancer) context.

The analysis included:
- Data cleaning and preprocessing.
- Handling class imbalance using resampling techniques.
- Hyperparameter optimization using Random Search and Grid Search.
- Evaluation using multiple performance metrics: Precision, Recall, Specificity, F1-Score, and AUC-ROC.
- Feature importance analysis for interpretability.
The goal was to select and fine-tune the best model based on the needs of a cancer diagnosis setting, where balancing precision and recall is critical.

Files:
1. Submission 1 Final.ipynb | First part of the project: Data preprocessing, model training, hyperparameter tuning, and evaluation using Random Forest and Gradient Boosting.
2. Submission 2 Final.ipynb | Final part: Extended model evaluation (ROC curves, F1-Scores, Feature Importance), interpretation of results, and selection of the best model.

### Libraries
```
#Call libraries
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns

#For balancing methods
from collections import Counter
from imblearn.over_sampling import SMOTE #SMOTE
from sklearn.datasets import make_classification #RUS
from imblearn.under_sampling import RandomUnderSampler #RUS
```
```
#Machine learning
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.model_selection import cross_val_score, RepeatedKFold #Cross Validation
from sklearn.metrics import confusion_matrix #Confusion matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    precision_score, roc_auc_score, roc_curve
)

#Statistical Analysis
from scipy.stats import t, ttest_rel
from scipy.stats import chi2, norm, shapiro, probplot

#Hyperparameter optimization
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

#Special algorithms
!pip install xgboost
!pip install catboost
```
#### XGBoost
```
import xgboost as xgb
```
#### Cat Boost
```
from catboost import CatBoostClassifier
```
#### Random Forests
```
from sklearn.ensemble import RandomForestClassifier
```
#### Gradient Boosting
```
from sklearn.ensemble import GradientBoostingClassifier
```
