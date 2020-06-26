# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/preparation//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import seaborn as sns
import datasist.project as dp
import datasist as ds

data = pd.read_csv('C:/Users/AIMITON/LOBA/Projects/comedy_rating/comedy-rating/data/raw/train.csv')
data.head()

data.info()

X=data.drop('Rating',axis=1)

y=data['Rating']

#Encode all categorical feature with label encoding
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
for col in X.columns:
    X[col] = lb.fit_transform(X[col])
X.head(10)


#export the processed data and label to the processed folder
dp.save_data(X, 'train_proc',method='csv', loc='raw')
dp.save_data(y, 'train_labels',method='csv', loc='raw')

'train_proc.csv'


