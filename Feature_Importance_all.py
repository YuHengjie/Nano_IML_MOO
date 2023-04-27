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
file_list = os.listdir('./Model_RDW')
xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

df_list = []

for i in range(1,11,1):
    exec("temp_pd = pd.read_excel(\"./Model_RDW/rules_{}.xlsx\",index_col = 0,)".format(i))
    temp_pd['Model'] = i
    df_list.append(temp_pd)

rules_all = pd.concat(df_list)
rules_all = rules_all.reset_index()
rules_all.to_excel('RuleFit_all_rules.xlsx')
rules_all = rules_all.drop(columns=['index'])
rules_all

# %%
print("Total number of generated rules: ",str(rules_all.shape[0]))

# %%
def find_mk(input_vars:list, rule:str):

    var_count = 0
    for var in input_vars:
        if var in rule:
            var_count += 1
    return(var_count)

def get_feature_importance(feature_set: list, rule_set: pd.DataFrame, scaled = False):

    feature_imp = list()
    
    rule_feature_count = rule_set.rule.apply(lambda x: find_mk(feature_set, x))

    for feature in feature_set:
        
        # find subset of rules that apply to a feature
        feature_rk = rule_set.rule.apply(lambda x: feature in x)
        
        # find importance of linear features
        linear_imp = rule_set[(rule_set.type=='linear')&(rule_set.rule==feature)].importance.values
        
        # find the importance of rules that contain feature
        rule_imp = rule_set.importance[(rule_set.type=='rule')&feature_rk]
        
        # find the number of features in each rule that contain feature
        m_k = rule_feature_count[(rule_set.type=='rule')&feature_rk]
        
        # sum the linear and rule importances, divided by m_k
        if len(linear_imp)==0:
            linear_imp = 0
        # sum the linear and rule importances, divided by m_k
        if len(rule_imp) == 0:
            feature_imp.append(float(linear_imp))
        else:
            feature_imp.append(float(linear_imp + (rule_imp/m_k).sum()))
        
    if scaled:
        feature_imp = 100*(feature_imp/np.array(feature_imp).max())
    
    return(feature_imp)

# %%
feature_names = ['Composition', 'Concentration (mg/L)', 'TEM size (nm)', 'Morphology',
       'Hydrodynamic diameter (nm)', 'Zeta potential (mV)',
       'BET surface area (m2/g)']
feature_importances = get_feature_importance(feature_names, rules_all, scaled=False)
importance_df = pd.DataFrame(feature_importances, index = feature_names, columns = ['importance']).sort_values(by='importance',ascending=False)
importance_df['importance'] = importance_df['importance']/10
importance_df

# %%
feature_color_list = ['#FCFEA4','#F7D13C','#FB9B06','#ED6825','#CF4446','#A42C60','#781C6D','#4A0B6A','#1A0B40']

color_list = ['#FCFEA4','#F7D13C','#FB9B06','#ED6825','#CF4446','#A42C60','#781C6D',]
fig, ax1 = plt.subplots(figsize=(4,2.2)) 

plt.bar(importance_df.index,importance_df['importance'],width=0.4,color=color_list,edgecolor='k',alpha=0.6)

plt.xticks(rotation=45,ha='right')
plt.ylim(0,3.3)
plt.xlim(-0.5,6.5)
plt.ylabel('Average feature importance')
plt.title('RDW',fontsize=12)
fig.savefig("./Image/Feature_importanca_RDW.jpg",dpi=600,bbox_inches='tight')

# %%
file_list = os.listdir('./Model_RMC')
xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

df_list = []

for i in range(1,11,1):
    exec("temp_pd = pd.read_excel(\"./Model_RMC/rules_{}.xlsx\",index_col = 0,)".format(i))
    temp_pd['Model'] = i
    df_list.append(temp_pd)

rules_all = pd.concat(df_list)
rules_all = rules_all.reset_index()
rules_all.to_excel('RuleFit_all_rules.xlsx')
rules_all = rules_all.drop(columns=['index'])
rules_all

# %%
print("Total number of generated rules: ",str(rules_all.shape[0]))

# %%
feature_names = ['Composition', 'Concentration (mg/L)', 'TEM size (nm)', 'Morphology',
       'Hydrodynamic diameter (nm)', 'Zeta potential (mV)',
       'BET surface area (m2/g)']
feature_importances = get_feature_importance(feature_names, rules_all, scaled=False)
importance_df = pd.DataFrame(feature_importances, index = feature_names, columns = ['importance']).sort_values(by='importance',ascending=False)
importance_df['importance'] = importance_df['importance']/10
importance_df


# %%
color_list = ['#FCFEA4','#F7D13C','#FB9B06','#ED6825','#CF4446','#A42C60','#781C6D',]

plt.style.use('classic')

fig, ax1 = plt.subplots(figsize=(4,2.2)) 
plt.bar(importance_df.index,importance_df['importance'],width=0.4,color=color_list,edgecolor='k',alpha=0.6)

plt.xticks(rotation=45,ha='right')
plt.ylim(0,1.8)
plt.xlim(-0.5,6.5)
plt.title('RMC',fontsize=12)

plt.ylabel('Average feature importance')
fig.savefig("./Image/Feature_importanca_RMC.jpg",dpi=600,bbox_inches='tight')

# %%