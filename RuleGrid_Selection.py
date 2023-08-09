# %%
import pandas as pd
import os
import re

import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier

from sklearn import metrics
from imodels import RuleFitClassifier

import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %%
data_generated = pd.read_excel("./Generated_data.xlsx",index_col = 0,)
data_generated

# %%
C_range = '(37.5,75.0]'
Z_range = '[-32.77,-11.6]'
B_range = '[4.07,56.85]'
H_range = '[344.45,933.73]'
T_range = '(24.155, 132.11]'

# %%
data_candidate = data_generated.copy()

data_candidate = data_candidate[(data_candidate['Concentration (mg/L)']>37.5) & (data_candidate['Concentration (mg/L)']<=75)]
data_candidate = data_candidate[(data_candidate['Zeta potential (mV)']>=-32.77) & (data_candidate['Zeta potential (mV)']<=-11.318)]
data_candidate = data_candidate[(data_candidate['BET surface area (m2/g)']>=4.07) & (data_candidate['BET surface area (m2/g)']<=43.93)]
data_candidate = data_candidate[(data_candidate['Hydrodynamic diameter (nm)']>=357.13) & (data_candidate['Hydrodynamic diameter (nm)']<=933.73)]
data_candidate = data_candidate[(data_candidate['TEM size (nm)']>24.155) & (data_candidate['TEM size (nm)']<=132.11)]

data_candidate

# %%
RuleGrid_49_index = data_candidate.index
np.save("RuleGrid_49_index.npy", RuleGrid_49_index)


# %%