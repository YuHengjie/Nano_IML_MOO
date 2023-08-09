# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.model_selection import StratifiedShuffleSplit 

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from imodels import RuleFitClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier
import random

import shap

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import warnings
warnings.filterwarnings("ignore")

# %%
# %%
X_new = pd.read_excel("./Generated_data.xlsx",index_col = 0,)
X_new

# %%
X_new = X_new.drop(columns=['Composition']) 
X_new

# %%

X_new['Seedling part 1'] = 1
X_new['Seedling part 2'] = 2
X_new['Seedling part 3'] = 3
X_new

# %%
model_paras_RDW = pd.read_excel("./RuleFit_parameters_RDW.xlsx",index_col = 0,)
model_paras_RMC = pd.read_excel("./RuleFit_parameters_RMC.xlsx",index_col = 0,)
model_paras_RMC

# %% 
data_RDW = pd.read_excel("./Dataset_RDW.xlsx",index_col = 0,)
data_RMC = pd.read_excel("./Dataset_RMC.xlsx",index_col = 0,)
data_RMC

# %% 
X_RDW = data_RDW.drop(columns=['RDW','Solubility','Composition']) 
X_RMC = data_RMC.drop(columns=['RMC','Solubility']) 

# %%
le_composition = LabelEncoder()
le_composition.fit(X_RMC['Composition'])
X_RMC['Composition'] = le_composition.transform(X_RMC['Composition'])
print(list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])))

X_RMC['Seedling part'] = X_RMC['Seedling part'].map({'Stem':1,'Leaf':2,'Shoot':3})
X_RMC

# %%
target_name = 'RDW'
y_RDW = data_RDW.loc[:,target_name]
y_RDW

# %%
target_name = 'RMC'
y_RMC = data_RMC.loc[:,target_name]
y_RMC

# %%
feature_names = X_RDW.columns
feature_names

# %%
y_NPI_predict = pd.DataFrame(columns=range(1,11),index=X_new.index)

# for test
y_RDW_predict = pd.DataFrame(columns=range(1,11),index=X_new.index)
y_RMC_predict = pd.DataFrame(columns=range(1,11),index=X_new.index)
y_RMC1_predict = pd.DataFrame(columns=range(1,11),index=X_new.index)
y_RMC2_predict = pd.DataFrame(columns=range(1,11),index=X_new.index)
y_RMC3_predict = pd.DataFrame(columns=range(1,11),index=X_new.index)

for i in range(1,11,1):

    print('Processing: ',i)
    # RDW
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=i) 

    for train,test in sss.split(X_RDW, y_RDW):
        X_cv_RDW = X_RDW.iloc[train]
        y_cv_RDW = y_RDW.iloc[train]
        X_test_RDW = X_RDW.iloc[test]
        y_test_RDW = y_RDW.iloc[test]

    #if model_paras_RDW.loc[i,'tree_generator'] == 'None':
    model_RDW = RuleFitClassifier(random_state=42,include_linear=False,max_rules=model_paras_RDW.loc[i,'max_rules'],
                              n_estimators=model_paras_RDW.loc[i,'n_estimators'],tree_generator=eval(model_paras_RDW.loc[i,'tree_generator']),
                              tree_size=model_paras_RDW.loc[i,'tree_size'])

    model_RDW.fit(X_cv_RDW,y_cv_RDW)
    
    y_pred_cv = model_RDW.predict(X_cv_RDW)
    y_proba_cv = model_RDW.predict_proba(X_cv_RDW)[:, 1]

    y_pred = model_RDW.predict(X_test_RDW)
    y_proba = model_RDW.predict_proba(X_test_RDW)[:, 1]

    # test to see if the model is same with grid search 
    print('Train AUC (RDW): %.2f'%metrics.roc_auc_score(y_cv_RDW,y_proba_cv))
    print('Test AUC (RDW): %.2f'%metrics.roc_auc_score(y_test_RDW,y_proba))


    # RMC
    train_all = []
    test_all = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=i) 
    for index in [0,1,3,4,5]: # for composition without Fe element
        X_composition = X_RMC[X_RMC['Composition']==index]
        y_composition = y_RMC[X_RMC['Composition']==index]
        for train,test in sss.split(X_composition, y_composition):
            train_all.extend(list(X_composition.iloc[train,:].index))
            test_all.extend(list(X_composition.iloc[test,:].index))

    X_composition = X_RMC[X_RMC['Composition'].isin([2,6,7])] # for composition with Fe element
    y_composition = y_RMC[X_RMC['Composition'].isin([2,6,7])]
    for train,test in sss.split(X_composition, y_composition):
        train_all.extend(list(X_composition.iloc[train,:].index))
        test_all.extend(list(X_composition.iloc[test,:].index))


    X_cv_RMC = X_RMC.iloc[train_all]
    X_cv_RMC = X_cv_RMC.drop(columns=['Composition'])
    y_cv_RMC = y_RMC.iloc[X_cv_RMC.index]

    X_test_RMC = X_RMC.iloc[test_all]
    X_test_RMC = X_test_RMC.drop(columns=['Composition'])
    y_test_RMC = y_RMC.iloc[X_test_RMC.index]
    
    #if model_paras.loc[i,'tree_generator'] == 'None':
    model_RMC = RuleFitClassifier(random_state=42,include_linear=False,max_rules=model_paras_RMC.loc[i,'max_rules'],
                              n_estimators=model_paras_RMC.loc[i,'n_estimators'],tree_generator=eval(model_paras_RMC.loc[i,'tree_generator']),
                              tree_size=model_paras_RMC.loc[i,'tree_size'])

    model_RMC.fit(X_cv_RMC,y_cv_RMC)
    
    y_pred_cv = model_RMC.predict(X_cv_RMC)
    y_proba_cv = model_RMC.predict_proba(X_cv_RMC)[:, 1]

    y_pred = model_RMC.predict(X_test_RMC)
    y_proba = model_RMC.predict_proba(X_test_RMC)[:, 1]

    # test to see if the model is same with grid search 
    print('Train AUC (RMC): %.2f'%metrics.roc_auc_score(y_cv_RMC,y_proba_cv))
    print('Test AUC (RMC): %.2f'%metrics.roc_auc_score(y_test_RMC,y_proba))

    y_NPI_predict[y_NPI_predict.columns[i-1]] = 0.5 * (model_RDW.predict_proba(X_new.iloc[:,[*range(5)]])[:, 1] + np.mean([model_RMC.predict_proba(X_new.iloc[:,[*range(5),5]])[:, 0],model_RMC.predict_proba(X_new.iloc[:,[*range(5),6]])[:, 0],model_RMC.predict_proba(X_new.iloc[:,[*range(5),7]])[:, 0]], axis=0))
    y_RDW_predict[y_NPI_predict.columns[i-1]] = model_RDW.predict_proba(X_new.iloc[:,[*range(5)]])[:, 1]
    y_RMC_predict[y_NPI_predict.columns[i-1]] = np.mean([model_RMC.predict_proba(X_new.iloc[:,[*range(5),5]])[:, 1],model_RMC.predict_proba(X_new.iloc[:,[*range(5),6]])[:, 1],model_RMC.predict_proba(X_new.iloc[:,[*range(5),7]])[:, 1]], axis=0)
    y_RMC1_predict[y_NPI_predict.columns[i-1]] = model_RMC.predict_proba(X_new.iloc[:,[*range(5),5]])[:, 1]
    y_RMC2_predict[y_NPI_predict.columns[i-1]] = model_RMC.predict_proba(X_new.iloc[:,[*range(5),6]])[:, 1]
    y_RMC3_predict[y_NPI_predict.columns[i-1]] = model_RMC.predict_proba(X_new.iloc[:,[*range(5),7]])[:, 1]

    if i == 10:
        y_NPI_predict['Mean'] = y_NPI_predict.mean(axis=1)
        y_RDW_predict['Mean'] = y_RDW_predict.mean(axis=1)
        y_RMC_predict['Mean'] = y_RMC_predict.mean(axis=1)
        y_RMC1_predict['Mean'] = y_RMC1_predict.mean(axis=1)
        y_RMC2_predict['Mean'] = y_RMC2_predict.mean(axis=1)
        y_RMC3_predict['Mean'] = y_RMC3_predict.mean(axis=1)

        y_NPI_predict.to_excel("./SHAP_values/y_NPI_predict_"+str(i)+".xlsx")
        y_RDW_predict.to_excel("./SHAP_values/y_RDW_predict_"+str(i)+".xlsx")
        y_RMC_predict.to_excel("./SHAP_values/y_RMC_predict_"+str(i)+".xlsx")
        y_RMC1_predict.to_excel("./SHAP_values/y_RMC1_predict_"+str(i)+".xlsx")
        y_RMC2_predict.to_excel("./SHAP_values/y_RMC2_predict_"+str(i)+".xlsx")
        y_RMC3_predict.to_excel("./SHAP_values/y_RMC3_predict_"+str(i)+".xlsx")
    
    model_fn = lambda x: 0.5 * (model_RDW.predict_proba(x.iloc[:, [*range(5)]])[:, 1] + np.mean([model_RMC.predict_proba(x.iloc[:, [*range(5),5]])[:, 0], model_RMC.predict_proba(x.iloc[:, [*range(5),6]])[:, 0], model_RMC.predict_proba(x.iloc[:, [*range(5),7]])[:, 0]], axis=0))
    explainer = shap.Explainer(model_fn,X_new)
    shap_values = explainer(X_new)
    np.save("./SHAP_values/SHAP_values_"+str(i)+".npy", shap_values)

