# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/modeling//py
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

import datasist.project as dp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
#retrieve data from the processed folder
data = pd('C:\Users\AIMITON\LOBA\Projects\comedy_rating\comedy-rating\data\processed/')
label = dp.get_data("train_labels.csv", method='csv')


#base model with random forest
rf = RandomForestRegressor(n_estimators=10, random_state=2)
score = cross_val_score(estimator=rf,X=data, y=label.Rating, cv=5, scoring="neg_mean_squared_error",n_jobs=-1)
score = -1 * np.mean(score)
print("RMSE is {}".format(score))
#save the model
dp.save_model(rf, name='rf_model_n10')
# save the result
result = {"rmse_rf_model_n10": score}
dp.save_outputs(result,name='rmse_rf_model_n10')


