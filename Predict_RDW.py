# %%
import pandas as pd
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
X_new = pd.read_excel("./Generated_data.xlsx",index_col = 0,)
X_new

# %%
composition_dict = {'CeO2':0, 'CuO':1, 'Fe3O4':2, 'SiO2':3, 'TiO2':4, 'ZnO':5, 'α-Fe2O3':6, 'γ-Fe2O3':7}
le_composition = LabelEncoder()
le_composition.fit_transform(list(composition_dict.keys()))
X_new['Composition'] = le_composition.transform(X_new['Composition'])
print(list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])))

morpholog_dict = {'Compound':0, 'Spherical':1}
le_morphology = LabelEncoder()
le_morphology.fit_transform(list(morpholog_dict.keys()))
X_new['Morphology'] = le_morphology.transform(X_new['Morphology'])
print(list(le_morphology.inverse_transform([0,1])))

# %%
X_new = X_new.loc[:,['Concentration (mg/L)','Hydrodynamic diameter (nm)','BET surface area (m2/g)','Zeta potential (mV)','TEM size (nm)']]
X_new



# %%
model_paras = pd.read_excel("./RuleFit_parameters_RDW_simple.xlsx",index_col = 0,)
model_paras

# %%
# %% 
data = pd.read_excel("./dataset_RDW.xlsx",index_col = 0,)
data

# %% 
X = data.drop(columns=['RDW']) 

# %%
le_composition = LabelEncoder()
le_composition.fit(X['Composition'])
X['Composition'] = le_composition.transform(X['Composition'])
print(list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])))


le_morphology = LabelEncoder()
le_morphology.fit(X['Morphology'])
X['Morphology'] = le_morphology.transform(X['Morphology'])
print(list(le_morphology.inverse_transform([0,1])))
X

# %%
target_name = 'RDW'
y = data.loc[:,target_name]
y

# %%
feature_names = X.columns
feature_names

# %%

y_new_predict = pd.DataFrame(columns=range(1,11),index=X_new.index)

for i in range(1,11,1):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=i) 

    for train,test in sss.split(X, y):
        X_cv = X.iloc[train]
        y_cv = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]

    X_cv = X_cv.loc[:,['Concentration (mg/L)','Hydrodynamic diameter (nm)','BET surface area (m2/g)','Zeta potential (mV)','TEM size (nm)']]
    X_test = X_test.loc[:,['Concentration (mg/L)','Hydrodynamic diameter (nm)','BET surface area (m2/g)','Zeta potential (mV)','TEM size (nm)']]

    #if model_paras.loc[i,'tree_generator'] == 'None':
    model = RuleFitClassifier(random_state=42,include_linear=False,max_rules=model_paras.loc[i,'max_rules'],
                              n_estimators=model_paras.loc[i,'n_estimators'],tree_generator=eval(model_paras.loc[i,'tree_generator']),
                              tree_size=model_paras.loc[i,'tree_size'])

    model.fit(X_cv,y_cv)
    
    y_pred_cv = model.predict(X_cv)
    y_proba_cv = model.predict_proba(X_cv)[:, 1]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # test to see if the model is same with grid search 
    print('Train AUC: %.2f'%metrics.roc_auc_score(y_cv,y_proba_cv))
    print('Test AUC: %.2f'%metrics.roc_auc_score(y_test,y_proba))

    y_new_predict.loc[:,i] = model.predict_proba(X_new)[:, 1]

# %%
y_new_predict['Average'] = y_new_predict.mean(numeric_only=True, axis=1)
y_new_predict

# %%
y_new_predict.to_excel('Predict_RDW.xlsx')

# %%
