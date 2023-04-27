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
rules_all = rules_all[rules_all['importance']>0.1]
rules_all

# %%
rules_three = rules_all[rules_all['Feature number']==3]
rules_three

# %%
C_range = [25, 200]
H_range = [197, 933.73]
Z_range = [-32.77, 44.07]
T_range = [12.97, 132.11]
B_range = [4.07,200.84]

# set to 0 when first run, then change to the max value by manual
larger_coeff_limit = 0
larger_import_limit = 0

# %% the interested interval obtained from rule map of two features
C_two_range = '(37.5,75.0]'
H_two_range = '[220.665,933.73]'
Z_two_range = '[-32.77,-11.318]'
B_two_range = '[4.07,43.93]'

# %%
def interval_transfer (interval: str):
    left_endpoint, right_endpoint = interval.strip('()[]').split(',')
    close_status = ''

    if interval[0] == '(' and interval[-1] == ')':
        close_status = 'neither'
    if interval[0] == '[' and interval[-1] == ')':
        close_status = 'left'
    if interval[0] == '(' and interval[-1] == ']':
        close_status = 'right'
    if interval[0] == '[' and interval[-1] == ']':
        close_status = 'both'

    interval_return = pd.Interval(float(left_endpoint), float(right_endpoint), closed=close_status)
    return interval_return

# %%
C_two_range = interval_transfer(C_two_range)
H_two_range = interval_transfer(H_two_range)
Z_two_range = interval_transfer(Z_two_range)
B_two_range = interval_transfer(B_two_range)
B_two_range

# %% simple the intervals for better visualization, if several intervals are close
def simple_interval_func(intervals,threshold=10):
    interval_data = []
    for interval in intervals:
        left_paren, nums, right_paren = interval[0], interval[1:-1], interval[-1]
        num1, num2 = [float(num) for num in nums.split(', ')]
        interval_data.append([left_paren, num1, num2, right_paren])
    interval_df = pd.DataFrame(interval_data, columns=['left_paren', 'num1', 'num2', 'right_paren'])

    interval_df = interval_df.sort_values(by=['num1','num2'], ascending=True)
    interval_df['group'] = ''
    interval_df.iloc[0,4] = 0

    group_start = 0
    for index in range(1,interval_df.shape[0]):
        interval_df_temp = interval_df[interval_df['group']==group_start]
        if (abs(interval_df.iloc[index,1]-interval_df_temp['num1'].mean())<=threshold) and (abs(interval_df.iloc[index,2]-interval_df_temp['num2'].mean())<=threshold):
            interval_df.iloc[index,4] = group_start
        else:
            group_start += 1
            interval_df.iloc[index,4] = group_start

    interval_df['new num1'] = ''
    interval_df['new num2'] = ''

    for group_idnex in range(0,interval_df['group'].max()+1):
        interval_df_temp = interval_df[interval_df['group']==group_idnex]
        interval_df.loc[interval_df_temp.index,'new num1'] = round(interval_df_temp['num1'].mean(),3)
        interval_df.loc[interval_df_temp.index,'new num2'] = round(interval_df_temp['num2'].mean(),3)

    interval_df['simple'] = ''
    for index in interval_df.index:
        interval_df.loc[index,'simple'] = interval_df.loc[index,'left_paren']+str(interval_df.loc[index,'new num1'])+','+str(interval_df.loc[index,'new num2'])+interval_df.loc[index,'right_paren']
    
    interval_df=interval_df.sort_index(axis=0)

    return interval_df['simple'].values

# %% rule_to_interval_three
def rule_to_interval_three(matches: list, feature_1: str, feature_2: str, feature_3: str, feature_1_range: list, feature_2_range: list, feature_3_range: list): # order: 'B' > 'C' > 'R']

    interval_1 = 'None'
    interval_2 = 'None'
    interval_3 = 'None'

    feature_in = [matches[i][0] for i in range(len(matches))]

    feature_1_count = feature_in.count(feature_1)
    feature_2_count = feature_in.count(feature_2)
    feature_3_count = feature_in.count(feature_3)

    start = 0

    if feature_1_count > 0:
        if feature_1_count == 1:
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'
        else:
            if (matches[0][1] == '>') & (matches[1][1] == '<'):
                    interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>') & (matches[1][1] == '<='):
                    interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
            if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                    interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                    interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
        start += feature_1_count

    if feature_2_count > 0:
        if feature_2_count == 1:
            if matches[start+0][1] == '>':
                interval_2 = '(' + str(matches[start+0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[start+0][1] == '>=':
                interval_2 = '[' + str(matches[start+0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[start+0][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[start+0][2]) + ')'
            if matches[start+0][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[start+0][2]) + ']'
        else:
            if (matches[start+0][1] == '>') & (matches[start+1][1] == '<'):
                interval_2 = '(' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ')'
            if (matches[start+0][1] == '>') & (matches[start+1][1] == '<='):
                interval_2 = '(' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ']' 
            if (matches[start+0][1] == '>=') & (matches[start+1][1] == '<'):
                interval_2 = '[' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ')'
            if (matches[start+0][1] == '>=') & (matches[start+1][1] == '<='):
                interval_2 = '[' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ']' 
        start += feature_2_count

    if feature_3_count > 0:
        if feature_3_count == 1:
            if matches[start+0][1] == '>':
                interval_3 = '(' + str(matches[start+0][2]) + ', ' + str(feature_3_range[1]) + ']'
            if matches[start+0][1] == '>=':
                interval_3 = '[' + str(matches[start+0][2]) + ', ' + str(feature_3_range[1]) + ']'
            if matches[start+0][1] == '<':
                interval_3 = '[' + str(feature_3_range[0]) + ', ' + str(matches[start+0][2]) + ')'
            if matches[start+0][1] == '<=':
                interval_3 = '[' + str(feature_3_range[0]) + ', ' + str(matches[start+0][2]) + ']'
        else:
            if (matches[start+0][1] == '>') & (matches[start+1][1] == '<'):
                interval_3 = '(' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ')'
            if (matches[start+0][1] == '>') & (matches[start+1][1] == '<='):
                interval_3 = '(' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ']' 
            if (matches[start+0][1] == '>=') & (matches[start+1][1] == '<'):
                interval_3 = '[' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ')'
            if (matches[start+0][1] == '>=') & (matches[start+1][1] == '<='):
                interval_3 = '[' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ']' 

    return interval_1, interval_2, interval_3

# %% 1. rule map for B C T

# %% 
feature_1 = 'B'
feature_2 = 'C'
feature_3 = 'T'
feature_1_range = B_range
feature_2_range = C_range
feature_3_range = T_range

rule = 'B <= 150.0 and C <= 150.47 and T <= 29.325'
matches_raw = re.findall(r'(B|C|T)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_three(matches, feature_1, feature_2, feature_3, feature_1_range, feature_2_range, feature_3_range)

# %%
B_C_T_index = []

for i,rule in enumerate(rules_three['rule']):
    if ('B' in rule) and ('C' in rule) and ('T' in rule):
        B_C_T_index.append(i)

rule_B_C_T = rules_three.iloc[B_C_T_index,:]
rule_B_C_T = rule_B_C_T.reset_index(drop=True)
rule_B_C_T

# %%
rule_B_C_T['BET surface area (m2/g)'] = ''
rule_B_C_T['Concentration (mg/L)'] = ''
rule_B_C_T['TEM size (nm)'] = ''

for i,rule in enumerate(rule_B_C_T['rule']):
    matches_raw = re.findall(r'(B|C|T)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_B_C_T.loc[i,['BET surface area (m2/g)','Concentration (mg/L)','TEM size (nm)']] = rule_to_interval_three(matches, feature_1, feature_2, feature_3, feature_1_range, feature_2_range, feature_3_range)

rule_B_C_T



# %%
index_overlap_B_C_T = []

for i in range(0,rule_B_C_T.shape[0]):
    overlap_status = 1
    if not B_two_range.overlaps(interval_transfer(rule_B_C_T.loc[i,'BET surface area (m2/g)'])):
        overlap_status = 0
    if not C_two_range.overlaps(interval_transfer(rule_B_C_T.loc[i,'Concentration (mg/L)'])):
        overlap_status = 0

    if overlap_status == 1:
        index_overlap_B_C_T.append(i)

rule_B_C_T_focus = rule_B_C_T.iloc[index_overlap_B_C_T,:]
rule_B_C_T_focus = rule_B_C_T_focus.reset_index(drop=True)
rule_B_C_T_focus


# %%
print(rule_B_C_T_focus.sort_values('BET surface area (m2/g)')['BET surface area (m2/g)'].unique())


# %%
rule_B_C_T_sort_B = { '[4.07, 45.705]':0, '[4.07, 79.075]':1,
                      '[4.07, 98.515]':2,'[4.07, 107.75]':3, 
                    }

# %%
print(rule_B_C_T_focus.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

# %%
rule_B_C_T_sort_C = { '[25, 75.0]':0,
                    }


# %%
print(rule_B_C_T_focus.sort_values('TEM size (nm)')['TEM size (nm)'].unique())

# %%
rule_B_C_T_sort_T = { '(16.785, 132.11]':0, '(19.155, 132.11]':1 ,'(24.155, 132.11]':2 ,
                    }

# %%
rule_B_C_T_simply = rule_B_C_T_focus.copy()
rule_B_C_T_combnination = rule_B_C_T_simply.copy()

for i in range(rule_B_C_T_combnination.shape[0]):
    rule_B_C_T_combnination.loc[i,'Combination'] = rule_B_C_T_combnination.loc[i,'BET surface area (m2/g)'] + ' and ' + rule_B_C_T_combnination.loc[i,'Concentration (mg/L)']

rule_B_C_T_combnination


# %%
rule_map_B_C_T_importance = pd.DataFrame(0,index=list(rule_B_C_T_combnination['Combination'].unique()),columns=list(rule_B_C_T_sort_T),)
rule_map_B_C_T_frequency = pd.DataFrame(0,index=list(rule_B_C_T_combnination['Combination'].unique()),columns=list(rule_B_C_T_sort_T),)
rule_map_B_C_T_coefficient = pd.DataFrame(0,index=list(rule_B_C_T_combnination['Combination'].unique()),columns=list(rule_B_C_T_sort_T),)
rule_map_B_C_T_importance

# %%
for i in range(rule_B_C_T_combnination.shape[0]):
    rule_map_B_C_T_importance.loc[rule_B_C_T_combnination.loc[i,'Combination'],rule_B_C_T_combnination.loc[i,'TEM size (nm)']] += rule_B_C_T_combnination.loc[i,'importance']/10
    rule_map_B_C_T_coefficient.loc[rule_B_C_T_combnination.loc[i,'Combination'],rule_B_C_T_combnination.loc[i,'TEM size (nm)']] += rule_B_C_T_combnination.loc[i,'coef']/10
    rule_map_B_C_T_frequency.loc[rule_B_C_T_combnination.loc[i,'Combination'],rule_B_C_T_combnination.loc[i,'TEM size (nm)']] += 1/10

rule_map_B_C_T_importance

# %%
rule_map_B_C_T_plot = pd.DataFrame(columns=['TEM size (nm)','Combination','Importance','Coefficient','Frequency'])
for Com_item in list(rule_B_C_T_combnination['Combination'].unique()):
    for T_item in  list(rule_B_C_T_sort_T):
        temp_rule_map = pd.DataFrame({'TEM size (nm)':T_item,
                                        'Combination':Com_item,
                                        'Importance': rule_map_B_C_T_importance.loc[Com_item,T_item],
                                        'Coefficient':rule_map_B_C_T_coefficient.loc[Com_item,T_item],
                                        'Frequency':rule_map_B_C_T_frequency.loc[Com_item,T_item],
                                        }, index=[0])
        rule_map_B_C_T_plot = pd.concat([rule_map_B_C_T_plot,temp_rule_map],ignore_index=True)
rule_map_B_C_T_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (2.5, 2))

palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_B_C_T_coefficient.min().min()), abs(rule_map_B_C_T_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_B_C_T_coefficient.min().min()), abs(rule_map_B_C_T_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_B_C_T_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_B_C_T_importance.max().max()
    print('Warning: get larger importance limit for plot!')

g = sns.scatterplot(
    data=rule_map_B_C_T_plot,x="Combination", y="TEM size (nm)", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(30,250*rule_map_B_C_T_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')

plt.xlim(-0.5,3.5)
plt.ylim(-0.5,2.5)

plt.legend(bbox_to_anchor=(3.2, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

plt.title('RMC')
plt.xlabel('BET surface area (m2/g) and Concentration (mg/L)')

norm = plt.Normalize(-larger_coeff_limit, larger_coeff_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.7, anchor=(0, 1.0))

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/Rule_Map_RMC_B_C_T.jpg",dpi=600,bbox_inches='tight')

# %%