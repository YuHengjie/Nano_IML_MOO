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
model_paras = pd.read_excel("./RuleFit_parameters_RDW.xlsx",index_col = 0,)
model_paras

# %% 
data = pd.read_excel("./Dataset_RDW.xlsx",index_col = 0,)
data

# %% 
X = data.drop(columns=['RDW','Solubility','Composition']) 
X

# %%
target_name = 'RDW'
y = data.loc[:,target_name]
y


# %%

for i in range(1,11,1):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=i) 

    for train,test in sss.split(X, y):
        X_cv = X.iloc[train]
        y_cv = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]

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

    X_test['Observed RDW'] = y_test
    X_test['Predicted RDW'] = y_pred
    X_test['Predicted Prob RDW'] = y_proba  
    X_test.to_excel('./Model_RDW/Predict_Prob'+'_'+str(i)+'.xlsx')

# %%














# %% RMC

# %%
model_paras = pd.read_excel("./RuleFit_parameters_RMC.xlsx",index_col = 0,)
model_paras

# %% 
data = pd.read_excel("./Dataset_RMC.xlsx",index_col = 0,)
data

# %% 
X = data.drop(columns=['RMC','Solubility']) #,'TEM size (nm)'
print(len(X.columns))
print(X.columns)

# %%
le_composition = LabelEncoder()
le_composition.fit(X['Composition'])
X['Composition'] = le_composition.transform(X['Composition'])
print(list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])))

X['Seedling part'] = X['Seedling part'].map({'Stem':1,'Leaf':2,'Shoot':3})
X

# %%
target_name = 'RMC'
y = data.loc[:,target_name]
y

# %%

for i in range(1,11,1):

    train_all = []
    test_all = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=i) 
    for index in [0,1,3,4,5]: # for composition without Fe element
        X_composition = X[X['Composition']==index]
        y_composition = y[X['Composition']==index]
        for train,test in sss.split(X_composition, y_composition):
            train_all.extend(list(X_composition.iloc[train,:].index))
            test_all.extend(list(X_composition.iloc[test,:].index))

    X_composition = X[X['Composition'].isin([2,6,7])] # for composition with Fe element
    y_composition = y[X['Composition'].isin([2,6,7])]
    for train,test in sss.split(X_composition, y_composition):
        train_all.extend(list(X_composition.iloc[train,:].index))
        test_all.extend(list(X_composition.iloc[test,:].index))


    X_cv = X.iloc[train_all]
    X_cv = X_cv.drop(columns=['Composition'])
    y_cv = y.iloc[X_cv.index]

    X_test = X.iloc[test_all]
    X_test = X_test.drop(columns=['Composition'])
    y_test = y.iloc[X_test.index]

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

    X_test['Observed RMC'] = y_test
    X_test['Predicted RMC'] = y_pred
    X_test['Predicted Prob RMC'] = y_proba  
    X_test.to_excel('./Model_RMC/Predict_Prob'+'_'+str(i)+'.xlsx')

# %%
