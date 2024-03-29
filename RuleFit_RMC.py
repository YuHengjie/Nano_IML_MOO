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
from sklearn.model_selection import GridSearchCV,LeaveOneOut
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import random
import sys

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

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
df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))

best_parameters = []

for random_seed in range(1,11,1):

    train_all = []
    test_all = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_seed) 
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

    parameter = {
         "n_estimators":range(100,700,100),
         "tree_generator":[None,RandomForestRegressor(),GradientBoostingRegressor(),GradientBoostingClassifier()],
         "tree_size":[3,4,5,6,7],
         "max_rules":range(26,54,2),     
    }

    cv = 10
    model_gs = RuleFitClassifier(random_state=42,include_linear=False,)

    grid_search = GridSearchCV(model_gs, param_grid = parameter, scoring='accuracy', cv=cv, n_jobs=-1)
    grid_search.fit(X_cv, y_cv)

    best_parameters.append(["Best parameters for "+str(random_seed)+": ",grid_search.best_params_,])

    RuleFit_GS_best = grid_search.best_estimator_
    
    y_pred_cv = RuleFit_GS_best.predict(X_cv)
    y_proba_cv = RuleFit_GS_best.predict_proba(X_cv)[:, 1]

    y_pred = RuleFit_GS_best.predict(X_test)
    y_proba = RuleFit_GS_best.predict_proba(X_test)[:, 1]
    
    df_result = df_result.append(pd.Series({'AUC_CV':sum(cross_val_score(RuleFit_GS_best, X_cv, y_cv, cv=cv, scoring='roc_auc'))/cv,
                                                'F1_CV':sum(cross_val_score(RuleFit_GS_best, X_cv, y_cv, cv=cv, scoring='f1_weighted'))/cv,
                                                'Accuracy_CV':sum(cross_val_score(RuleFit_GS_best, X_cv, y_cv, cv=cv, scoring='accuracy'))/cv,
                                                'Train AUC':metrics.roc_auc_score(y_cv,y_proba_cv),
                                                'Train F1':metrics.f1_score(y_cv,y_pred_cv,average='weighted'),
                                                'Train Accuracy':metrics.accuracy_score(y_cv,y_pred_cv),
                                                'Test AUC':metrics.roc_auc_score(y_test,y_proba),
                                                'Test F1':metrics.f1_score(y_test,y_pred,average='weighted'),
                                                'Test Accuracy':metrics.accuracy_score(y_test,y_pred)}),ignore_index=True)
    
    df_result.to_excel('./Model_RMC_5/GridSearch'+'_performance_'+str(random_seed)+'.xlsx')
    X_test['Observed RMC'] = y_test
    X_test['Predicted RMC'] = y_pred
    X_test.to_excel('./Model_RMC_5/Predict_observe'+'_'+str(random_seed)+'.xlsx')

    rules = RuleFit_GS_best.get_rules()
    rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
    rules.to_excel("./Model_RMC_5/rules_"+str(random_seed)+".xlsx")

    list_str = '\n'.join(str(item) for item in best_parameters)
    # Write string to file
    with open('./Model_RMC_5/GridSearch_parameters_{}.txt'.format(random_seed), 'w') as file:
        file.write(list_str)