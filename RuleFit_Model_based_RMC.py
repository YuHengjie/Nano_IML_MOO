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

import itertools

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %%
file_list = os.listdir('./Model_RMC_simple')
xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

df_list = []

for i in range(1,11,1):
    exec("temp_pd = pd.read_excel(\"./Model_RMC_simple/rules_{}.xlsx\",index_col = 0,)".format(i))
    temp_pd['Model'] = i
    df_list.append(temp_pd)

rules_all = pd.concat(df_list)
rules_all = rules_all.reset_index()
rules_all = rules_all.drop(columns=['index'])
rules_all

# %%
print("Total number of generated rules: ",str(rules_all.shape[0]))

# %%
model_rule_number_RMC = pd.DataFrame(index=range(1,11),columns=['Rule number'])
for i in range(1,11):
    model_rule_number_RMC.loc[i,'Rule number'] = rules_all[rules_all['Model']==i].shape[0]
model_rule_number_RMC

# %%
model_rule_number_RMC.to_excel('Rule_number_RMC_simple.xlsx')

# %%
model_rule_number_RDW = pd.read_excel('Rule_number_RDW_simple.xlsx',index_col=0)
model_rule_number_RDW

# %%

fig=plt.figure(figsize=(5,3.5))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.bar(model_rule_number_RDW.index,model_rule_number_RDW['Rule number'],color='#E9D3C5')

ax2.bar(model_rule_number_RMC.index,model_rule_number_RMC['Rule number'],color='#BA768F')

ax1.get_shared_x_axes().join(ax1, ax2)

ax1.set_ylabel('Rule number (RDW)')
ax2.set_ylabel('Rule number (RMC)')

ax2.set_xlabel('Model')

ax1.set_xticks(range(1,11))
ax1.set_xticklabels([])
ax2.set_xticks(range(1,11))

fig.savefig("./Image/RuleFit_rule_number_models_simple.jpg",dpi=600,bbox_inches='tight')

# %%
rules_all.to_excel('Rules_all_RMC_simple.xlsx')

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
feature_names = ['Concentration (mg/L)', 'TEM size (nm)',
       'Hydrodynamic diameter (nm)', 'Zeta potential (mV)',
       'BET surface area (m2/g)']
feature_importances = get_feature_importance(feature_names, rules_all, scaled=False)
importance_df = pd.DataFrame(feature_importances, index = feature_names, columns = ['importance']).sort_values(by='importance',ascending=False)
importance_df['importance'] = importance_df['importance']/10
importance_df

# %%
color_list = ['#FCFEA4','#FB9B06','#ED6825','#A42C60','#4A0B6A']
fig, ax = plt.subplots(figsize=(3.5,2.7)) 
plt.style.use('classic')
plt.bar(importance_df.index,importance_df['importance'],width=0.4,color=color_list,edgecolor='k',alpha=0.6)

plt.xticks(rotation=45,ha='right')
plt.ylim(0,3.3)
plt.xlim(-0.5,4.5)
plt.ylabel('Average feature importance')

fig.savefig("./Image/RuleFit_importance_feature_RMC_simple.jpg",dpi=600,bbox_inches='tight')

# %%
importance_df['Model'] = 'RMC'
importance_df.to_excel('RMC_feature_importance_simple.xlsx')
importance_df

# %%
importance_df_two_model = pd.read_excel('RDW_feature_importance_simple.xlsx',index_col=0)
importance_df_two_model

# %%
importance_df_two_model = pd.concat([importance_df_two_model,importance_df])
importance_df_two_model['Feature'] = importance_df_two_model.index
importance_df_two_model = importance_df_two_model.reset_index(drop=True)
importance_df_two_model


# %%
palette = ['#E9D3C5','#BA768F',]

figure = plt.figure(figsize=(4,2.7))
plt.style.use('classic')
bar = sns.barplot(data=importance_df_two_model, x="Feature", y="importance",hue='Model', 
                  palette=sns.color_palette(palette, 2),edgecolor='w')

bar.set_xlabel('')
bar.set_ylabel('Average feature importance')

plt.ylim([0,3.3])

leg = plt.legend(loc=[0.6,0.7], frameon=False)

sns.move_legend(bar, "upper right", fontsize=12)

plt.xticks(rotation=45,ha='right')

figure.savefig("./Image/RuleFit_feature_importance_two_models.jpg",dpi=600,bbox_inches='tight')

# %%
feature_dict = {'C': 'Concentration (mg/L)',
           'Z': 'Zeta potential (mV)',
           'H': 'Hydrodynamic diameter (nm)',
           'T': 'TEM size (nm)',
           'B': 'BET surface area (m2/g)'}
feature_dict

# %% simplify the rules
rules_all['rule'] = rules_all['rule'].str.replace(feature_dict['C'],'C',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace(feature_dict['Z'],'Z',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace(feature_dict['H'],'H',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace(feature_dict['T'],'T',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace(feature_dict['B'],'B',regex=False)
rules_all

# %%
def count_feature_letters(s):
    letters = set(['C', 'Z', 'H', 'T', 'B'])
    return len(set(filter(lambda c: c in letters, s)))

# %%
rules_all['Feature number'] = rules_all['rule'].apply(count_feature_letters)
rules_all

# %%
rules_all['Feature number'].unique()

# %%
rule_number_importance = pd.DataFrame(columns=['Rule number','Importance'],index=range(3))
rule_number_importance.loc[0,:] = [1,rules_all[rules_all['Feature number']==1]['importance'].sum()/10]
rule_number_importance.loc[1,:] = [2,rules_all[rules_all['Feature number']==2]['importance'].sum()/10]
rule_number_importance.loc[2,:] = [3,rules_all[rules_all['Feature number']==3]['importance'].sum()/10]
rule_number_importance.loc[3,:] = [4,rules_all[rules_all['Feature number']==4]['importance'].sum()/10]
rule_number_importance

# %%
color_list = ['#FCFEA4','#ED6825','#A42C60','#4A0B6A']

fig, ax1 = plt.subplots(figsize=(2.8,2.4)) 
plt.style.use('classic')

plt.bar(rule_number_importance['Rule number'],rule_number_importance['Importance'],
        width=0.4,color=color_list,edgecolor='k',alpha=0.6)

plt.xlabel('Feature number')
plt.xticks([1,2,3,4])
plt.ylim(0,4.9)
plt.ylabel('Average rule importance')

fig.savefig("./Image/RuleFit_number_importance_RMC.jpg",dpi=600,bbox_inches='tight')

# %%
rule_number_importance['Model'] = 'RMC'
rule_number_importance.to_excel('RMC_number_importance_simple.xlsx')
rule_number_importance

# %%
importance_number_two_model = pd.read_excel('RDW_number_importance_simple.xlsx',index_col=0)
importance_number_two_model

# %%
importance_number_two_model = pd.concat([importance_number_two_model,rule_number_importance])
importance_number_two_model = importance_number_two_model.reset_index(drop=True)
importance_number_two_model

# %%
palette = ['#E9D3C5','#BA768F',]

figure = plt.figure(figsize=(3.5,2.4))
plt.style.use('classic')
bar = sns.barplot(data=importance_number_two_model, x="Rule number", y="Importance",hue='Model', 
                  palette=sns.color_palette(palette, 2),edgecolor='w')

bar.set_ylabel('Average rule importance')
bar.set_xlabel('Feature number')
plt.ylim([0,4.8])

leg = plt.legend(loc=[0.6,0.7], frameon=False)

sns.move_legend(bar, "upper right", fontsize=12)

figure.savefig("./Image/RuleFit_number_importance_two_models.jpg",dpi=600,bbox_inches='tight')

# %%
rules_all_one = rules_all[rules_all['Feature number']==1]
rules_all_one

# %%
rules_all_two = rules_all[rules_all['Feature number']==2]
rules_all_two

# %%
letters = feature_dict.keys()
combinations = list(itertools.combinations(letters, 2))
col1, col2 = zip(*combinations)
rules_two_importance = pd.DataFrame({'Feature 1': col1, 'Feature 2': col2})
rules_two_importance

# %%
for i in range(0,rules_two_importance.shape[0]):
    index = []
    for j in range(0,rules_all_two.shape[0]):
        if (rules_two_importance.loc[i,'Feature 1'] in rules_all_two.iloc[j,0]) and (rules_two_importance.loc[i,'Feature 2'] in rules_all_two.iloc[j,0]):
            index.append(j)
    rules_two_importance.loc[i,'Importance'] = rules_all_two.iloc[index,:]['importance'].sum()/10
rules_two_importance

# %%
rules_two_importance['Importance'].sum()

# %%
rules_all_four = rules_all[rules_all['Feature number']==4]
rules_all_four

# %%
rules_four_importance = pd.DataFrame({'Feature 1': feature_dict.keys(), 'Feature 2':'~None'})
rules_four_importance

# %%
for i in range(0,rules_four_importance.shape[0]):
    index = []
    for j in range(0,rules_all_four.shape[0]):
        if rules_four_importance.loc[i,'Feature 1'] not in rules_all_four.iloc[j,0]:
            index.append(j)
    rules_four_importance.loc[i,'Importance'] = rules_all_four.iloc[index,:]['importance'].sum()/10
rules_four_importance

# %%
rules_four_importance['Importance'].sum()

# %%
rules_one_two_four_importance = pd.concat([rules_two_importance,rules_four_importance])
rules_one_two_four_importance = rules_one_two_four_importance.reset_index(drop=True)
rules_one_two_four_importance

# %%
for i in range(0,rules_one_two_four_importance.shape[0]):
    rules_one_two_four_importance.loc[i,'Feature 1'] = feature_dict[rules_one_two_four_importance.loc[i,'Feature 1']]
    if rules_one_two_four_importance.loc[i,'Feature 2'] not in ['None','~None']:
        rules_one_two_four_importance.loc[i,'Feature 2'] = feature_dict[rules_one_two_four_importance.loc[i,'Feature 2']]
rules_one_two_four_importance

# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (2, 2))

larger_limit_one_two_four = max(rules_one_two_four_importance['Importance'])


g = sns.scatterplot(
    data=rules_one_two_four_importance,x="Feature 1", y="Feature 2", hue="Importance",linewidth=0.2,
    s=150,ax=ax, palette = 'ch:s=-.1,r=.5',hue_norm=(0,larger_limit_one_two_four*1.05),
)
plt.xticks(rotation=45,ha='right')

plt.xlim(-0.5,4.5)
plt.ylim(-0.5,4.5)

plt.xlabel('')
plt.ylabel('')

plt.title('RMC: containing 2 and 4 features')

legend = g.legend(bbox_to_anchor=(1.5, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1, #title='Average rule importance',
          )

ax.text(7.3, 2.1, 'Average rule importance', rotation=270, va='center', ha='center')

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/RuleFit_importance_one_two_four_RMC.jpg",dpi=600,bbox_inches='tight')

# %%
rules_all_three = rules_all[rules_all['Feature number']==3]
rules_all_three

# %%
letters = feature_dict.keys()
combinations = list(itertools.combinations(letters, 3))
col1, col2, col3 = zip(*combinations)
rules_three_importance = pd.DataFrame({'Feature 1': col1, 'Feature 2': col2, 'Feature 3': col3})
rules_three_importance

# %%
for i in range(0,rules_three_importance.shape[0]):
    index = []
    for j in range(0,rules_all_three.shape[0]):
        if (rules_three_importance.loc[i,'Feature 1'] in rules_all_three.iloc[j,0]) and (rules_three_importance.loc[i,'Feature 2'] in rules_all_three.iloc[j,0])and (rules_three_importance.loc[i,'Feature 3'] in rules_all_three.iloc[j,0]):
            index.append(j)
    rules_three_importance.loc[i,'Importance'] = rules_all_three.iloc[index,:]['importance'].sum()/10
rules_three_importance

# %%
rules_three_importance['Importance'].sum()

# %%
for i in range(0,rules_three_importance.shape[0]):
    rules_three_importance.loc[i,'Feature 1'] = feature_dict[rules_three_importance.loc[i,'Feature 1']]
    rules_three_importance.loc[i,'Feature 2'] = feature_dict[rules_three_importance.loc[i,'Feature 2']]
    rules_three_importance.loc[i,'Feature 3'] = feature_dict[rules_three_importance.loc[i,'Feature 3']]
    rules_three_importance.loc[i,'Combination'] = rules_three_importance.loc[i,'Feature 1']+'\nand '+rules_three_importance.loc[i,'Feature 2']
rules_three_importance


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (4.5,1.5))

g = sns.scatterplot(
    data=rules_three_importance,x="Combination", y="Feature 3", hue="Importance",linewidth=0.2,
    s=175,ax=ax, palette = 'ch:s=-.1,r=.5',hue_norm=(0,larger_limit_one_two_four*1.05),
)
plt.xticks(rotation=45,ha='right')

plt.ylim(-0.5,2.5)
plt.xlim(-0.5,5.5)

plt.xlabel('')
plt.ylabel('')

plt.title('RMC: containing 3 features')

legend = g.legend(bbox_to_anchor=(1.22, 1.07), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1, #title='Average rule importance',
          )

ax.text(6.9, 1.2, 'Average rule importance', rotation=270, va='center', ha='center')

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/RuleFit_importance_three_RMC.jpg",dpi=600,bbox_inches='tight')

# %%
