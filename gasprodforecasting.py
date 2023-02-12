#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 11:32:24 2023

@author: Daniel Bazán
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor 
 
sns.set_theme()

#the data is loaded and then read into a dataframe
df = pd.read_csv("~/Documents/Datasets/weekly-gasoline.csv").loc[::-1].reset_index(drop=True)
df["Difference From Same Week Last Year"] = df["Difference From Same Week Last Year"].str.replace("'","").fillna(0)
df["Cumulative Difference"] = df["Cumulative Difference"].str.replace("'","").fillna(0)
df = df.astype({"Difference From Same Week Last Year":"int64", "Cumulative Difference":"int64"})

#the dataframes is sliced per year
df2020 = df[df["Fiscal Year"]==2020]
df2021 = df[df["Fiscal Year"]==2021]
df2022 = df[df["Fiscal Year"]==2022]
df2023 = df[df["Fiscal Year"]==2023]

sns.lineplot(data=df, x="Fiscal Year", y="Current Year Production")

plt.figure()
sns.lineplot(data=df, x="Fiscal Week", y="Current Year Production", hue="Fiscal Year", palette="flare")

#We can see that across all the three years there are repetitive patterns, with the exception
#of 2020, when the covid pandemic had become a global problem, these outliers could become very problematic
#to the machine training, I think that removing the data from 2020 should be very beneficial

plt.figure()
sns.lineplot(data=df, x="Fiscal Week", y="Current Year Production", palette="flare")

df_2020 = df.drop(df[df["Fiscal Year"]==2020].index)

plt.figure()
sns.lineplot(data=df_2020, x="Fiscal Week", y="Current Year Production", palette="flare")

plt.figure()
sns.lineplot(data=df_2020, x="Fiscal Week", y="Current Year Production", hue="Fiscal Year", palette="flare")

plt.figure()
sns.boxplot(data=df_2020, x="Fiscal Week", y="Current Year Production")
#we can see a decrease of production in the late/winter-spring weeks

#I´ll try to forecast the 2023 current year production, the training data will be the mean current year production

X_train = df_2020.drop(df_2020[df_2020["Fiscal Year"]==2023].index).drop("Current Year Production", axis=1)
y_train = df_2020["Current Year Production"].drop(df_2020[df_2020["Fiscal Year"]==2023].index)

plt.figure()
sns.heatmap(df_2020.corr(), cmap="flare", annot=True )

#The most correlated variables to current year production are the difference from same week last year and cumulative difference
#the fiscal year is almost completely uncorrelated

from sklearn.model_selection import GridSearchCV,  KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

xgb = XGBRegressor(verbosity = 1)

kf = KFold(10) 

X_test = df2023.drop("Current Year Production", axis=1)
y_test = df2023["Current Year Production"]

params = { "learning_rate": [.00001, .0001, .001, .01, .1], 
           "min_split_loss": [1, 35, 100],
           "max_depth": [3, 6, 10, 18, 32],
           "reg_lambda": [1, 3, 10],
           "reg_alpha": [.001, .1, 1, 3, 10]
           }

model = GridSearchCV(xgb, param_grid = params, cv=kf)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = model.score(X_test, y_test)
msre = mean_squared_error(y_test, y_pred)

plt.figure()
plt.plot(X_test["Fiscal Week"], y_test, color = "m")
plt.plot(X_test["Fiscal Week"], y_pred, color = "c")
plt.title("Forecasting")
plt.xlabel("Fiscal Weeks 2023")
plt.ylabel("Current Year Production")
plt.legend(["Actual 2023 Production", "Forecasted Production"])
plt.show()