# %%
import pandas as pd
import os
import re
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %%
file_list = os.listdir('./Model_RDW_simple')
xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

df_list = []

for i in range(1,11,1):
    exec("temp_pd = pd.read_excel(\"./Model_RDW_simple/rules_{}.xlsx\",index_col = 0,)".format(i))
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
rules_two = rules_all[rules_all['Feature number']==2]
rules_two

# %%
C_H_index = []
C_Z_index = []
C_T_index = []
C_B_index = []

for i,rule in enumerate(rules_two['rule']):
    if ('C' in rule) and ('H' in rule):
        C_H_index.append(i)
    if ('C' in rule) and ('Z' in rule):
        C_Z_index.append(i)
    if ('C' in rule) and ('T' in rule):
        C_T_index.append(i)
    if ('C' in rule) and ('B' in rule):
        C_B_index.append(i)

rule_C_H = rules_two.iloc[C_H_index,:]
rule_C_H = rule_C_H.reset_index(drop=True)

rule_C_Z = rules_two.iloc[C_Z_index,:]
rule_C_Z = rule_C_Z.reset_index(drop=True)

rule_C_T = rules_two.iloc[C_T_index,:]
rule_C_T = rule_C_T.reset_index(drop=True)

rule_C_B = rules_two.iloc[C_B_index,:]
rule_C_B = rule_C_B.reset_index(drop=True)

print(rule_C_H.shape[0],rule_C_Z.shape[0],rule_C_T.shape[0],rule_C_B.shape[0])

# %% the min and max value of each feature
C_range = [25, 200]
H_range = [197, 933.73]
Z_range = [-32.77, 44.07]
T_range = [12.97, 132.11]
B_range = [4.07,200.84]
# set to 0 when first run, then change to the max value by manual
larger_coeff_limit = 1.2639363307969915 
larger_import_limit = 0.32215302025759396 

# %% rule_to_interval_two
def rule_to_interval_two(matches: list, feature_1: str, feature_2: str, feature_1_range: list, feature_2_range: list): # order: 'B' > 'C' > 'R']

    interval_1 = 'None'
    interval_2 = 'None'

    if len(matches) == 1: # if only one statement, so only one feature
        # judge the feature, determin interval based on the logical operator and feature range
        if matches[0][0] == feature_1: 
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'
        if matches[0][0] == feature_2:
            if matches[0][1] == '>':
                interval_2 = '(' + str(matches[0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_2 = '[' + str(matches[0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[0][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[0][2]) + ']'

    if len(matches) == 2: # if the rule have two statements
        # judge the number of feature(s)
        if matches[0][0] == matches[1][0]:
            # if only one feature
            if matches[0][0] == feature_1:
                if (matches[0][1] == '>') & (matches[1][1] == '<'):
                        interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>') & (matches[1][1] == '<='):
                        interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
                if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                        interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                        interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'         
            if matches[0][0] == feature_2:
                if (matches[0][1] == '>') & (matches[1][1] == '<'):
                        interval_2 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>') & (matches[1][1] == '<='):
                        interval_2 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
                if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                        interval_2 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                        interval_2 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
        # if two feature
        else:
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'

            if matches[1][1] == '>':
                interval_2 = '(' + str(matches[1][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[1][1] == '>=':
                interval_2 = '[' + str(matches[1][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[1][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[1][2]) + ')'
            if matches[1][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[1][2]) + ']'

    if len(matches) == 3:
        # judge which feature appear twice
        if matches[0][0] == matches[1][0]:
            # the first feature appear twice
            if (matches[0][1] == '>') & (matches[1][1] == '<'):
                    interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>') & (matches[1][1] == '<='):
                    interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
            if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                    interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                    interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'  
            # so the second feature appear once       
            if matches[2][1] == '>':
                interval_2 = '(' + str(matches[2][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[2][1] == '>=':
                interval_2 = '[' + str(matches[2][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[2][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[2][2]) + ')'
            if matches[2][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[2][2]) + ']'

        else:
            # the second feature appear twice
            if (matches[1][1] == '>') & (matches[2][1] == '<'):
                    interval_2 = '(' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ')'
            if (matches[1][1] == '>') & (matches[2][1] == '<='):
                    interval_2 = '(' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ']' 
            if (matches[1][1] == '>=') & (matches[2][1] == '<'):
                    interval_2 = '[' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ')'
            if (matches[1][1] == '>=') & (matches[2][1] == '<='):
                    interval_2 = '[' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ']'  
            # so the first feature appear once  
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'

    if len(matches) == 4:
        # the first feature appear twice
        if (matches[0][1] == '>') & (matches[1][1] == '<'):
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
        if (matches[0][1] == '>') & (matches[1][1] == '<='):
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
        if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
        if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'    
        
        if (matches[2][1] == '>') & (matches[3][1] == '<'):
                interval_2 = '(' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ')'
        if (matches[2][1] == '>') & (matches[3][1] == '<='):
                interval_2 = '(' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ']' 
        if (matches[2][1] == '>=') & (matches[3][1] == '<'):
                interval_2 = '[' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ')'
        if (matches[2][1] == '>=') & (matches[3][1] == '<='):
                interval_2 = '[' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ']'  

    return interval_1, interval_2


# %% 1、 for rule map of C and H

# %% 
feature_1 = 'C'
feature_2 = 'H'
feature_1_range = C_range
feature_2_range = H_range

rule = 'C <= 75.0 and H <= 600 and C > 37.5'
matches_raw = re.findall(r'(C|H)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)


# %%
feature_1 = 'C'
feature_2 = 'H'
feature_1_range = C_range
feature_2_range = H_range

rule_C_H['Concentration (mg/L)'] = ''
rule_C_H['Hydrodynamic diameter (nm)'] = ''

for i,rule in enumerate(rule_C_H['rule']):
    matches_raw = re.findall(r'(C|H)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule) # 
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_C_H.loc[i,['Concentration (mg/L)','Hydrodynamic diameter (nm)']] = rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)

rule_C_H['Concentration (mg/L)'].astype(str)
rule_C_H['Hydrodynamic diameter (nm)'].astype(str)
rule_C_H


# %%
print(rule_C_H.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

# %%
rule_C_H_sort_C = {'[25, 37.5]':0,'[25, 75.0]':1,'[25, 150.0]':2,
                 '(37.5, 75.0]':3,'(37.5, 150.0]':4,'(37.5, 200]':5,
                 '(75.0, 150.0]':6, '(75.0, 200]':7,'(150.0, 200]':8,
       }

# %%

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

# %% test the function
original_list = ['[1, 1.065]', '[1, 1.105]','[1, 1.175]','(1.235, 1.465]', '(1.235, 1.87]', '(1.255, 1.87]',
                '(1.265, 1.395]', '(1.285, 1.395]', '(1.305, 1.395]', '(1.385, 1.645]',
                '(1.395, 1.87]', '(1.465, 1.49]',
                '[1, 1.235]', '[1, 1.255]', '[1, 1.3]','[1, 1.395]', 
                '[1, 1.505]', '[1, 1.635]','(1.055, 1.465]', '(1.065, 1.075]',
                '(1.065, 1.425]', '(1.065, 1.645]','(1.105, 1.645]', '(1.16, 1.87]',
                '(1.165, 1.87]', '(1.175, 1.515]',
                '(1.175, 1.645]', '(1.185, 1.295]', '(1.185, 1.635]', '(1.195, 1.295]',
                '(1.195, 1.645]', ]
interval_df = simple_interval_func(original_list,threshold=0.05)
interval_df

# %% simplify the interval for better visualization
print(rule_C_H.sort_values('Hydrodynamic diameter (nm)')['Hydrodynamic diameter (nm)'].unique())

# %%
C_H_interval_simply_H = pd.DataFrame({'Raw interval':['[197, 220.665]', '[197, 363.47]', '[197, 415.67999]',
                                      '(220.665, 363.47]', '(220.665, 543.41499]','(220.665, 840.285]', '(220.665, 933.73]', 
                                      '(263.34999, 344.45]', '(263.34999, 363.47]', '(263.34999, 415.67999]',
                                    '(263.34999, 933.73]', '(344.45, 900.76498]', '(363.47, 900.76498]',
                                    '(363.47, 933.73]', '(415.67999, 933.73]', '(900.76498, 933.73]',             
                                     ],})
C_H_interval_simply_H.loc[0:,'Simply interval'] = simple_interval_func(C_H_interval_simply_H['Raw interval'].values[0:],
                                                                       threshold=(H_range[1]-H_range[0])*0.05)

len(C_H_interval_simply_H['Simply interval'].unique())

# %%
C_H_interval_simply_H.to_excel('C_H_interval_simply_H.xlsx')
C_H_interval_simply_H

# %% replace the interval for better visualization
rule_C_H_simply = rule_C_H.copy()
for i,item in enumerate(rule_C_H_simply['Hydrodynamic diameter (nm)']):
     index = list(C_H_interval_simply_H['Raw interval']).index(item)
     rule_C_H_simply.loc[i,'Hydrodynamic diameter (nm)'] = C_H_interval_simply_H.loc[index,'Simply interval']
rule_C_H_simply

# %%
print(len(rule_C_H_simply['Hydrodynamic diameter (nm)'].unique()))
rule_C_H_sort_H = C_H_interval_simply_H['Simply interval'].unique()
rule_C_H_sort_H

# %%
rule_map_C_H_importance = pd.DataFrame(0,index=list(rule_C_H_sort_C.keys()),columns=list(rule_C_H_sort_H),)
rule_map_C_H_frequency = pd.DataFrame(0,index=list(rule_C_H_sort_C.keys()),columns=list(rule_C_H_sort_H),)
rule_map_C_H_coefficient = pd.DataFrame(0,index=list(rule_C_H_sort_C.keys()),columns=list(rule_C_H_sort_H),)
rule_map_C_H_importance

# %%
for i in range(rule_C_H_simply.shape[0]):
    rule_map_C_H_importance.loc[rule_C_H_simply.loc[i,'Concentration (mg/L)'],rule_C_H_simply.loc[i,'Hydrodynamic diameter (nm)']] += rule_C_H.loc[i,'importance']/10
    rule_map_C_H_coefficient.loc[rule_C_H_simply.loc[i,'Concentration (mg/L)'],rule_C_H_simply.loc[i,'Hydrodynamic diameter (nm)']] += rule_C_H.loc[i,'coef']/10
    rule_map_C_H_frequency.loc[rule_C_H_simply.loc[i,'Concentration (mg/L)'],rule_C_H_simply.loc[i,'Hydrodynamic diameter (nm)']] += 1/10

rule_map_C_H_importance

# %%
rule_map_C_H_plot = pd.DataFrame(columns=['Hydrodynamic diameter (nm)','Concentration (mg/L)','Importance','Coefficient','Frequency'])
for C_item in list(rule_C_H_sort_C.keys()):
    for H_item in  list(rule_C_H_sort_H):
        temp_rule_map = pd.DataFrame({'Hydrodynamic diameter (nm)':H_item,
                                        'Concentration (mg/L)':C_item,
                                        'Importance': rule_map_C_H_importance.loc[C_item,H_item],
                                        'Coefficient':rule_map_C_H_coefficient.loc[C_item,H_item],
                                        'Frequency':rule_map_C_H_frequency.loc[C_item,H_item],
                                        }, index=[0])
        rule_map_C_H_plot = pd.concat([rule_map_C_H_plot,temp_rule_map],ignore_index=True)
rule_map_C_H_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (3.5, 4.5))

palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_C_H_coefficient.min().min()), abs(rule_map_C_H_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_C_H_coefficient.min().min()), abs(rule_map_C_H_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_C_H_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_C_H_importance.max().max()
    print('Warning: get larger importance limit for plot!')


g = sns.scatterplot(
    data=rule_map_C_H_plot,x="Concentration (mg/L)", y="Hydrodynamic diameter (nm)", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(50,250*rule_map_C_H_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')
plt.xlim(-0.5,8.5)

plt.legend(bbox_to_anchor=(2, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

plt.title('RDW')

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

fig.savefig("./Image/Rule_Map_RDW_C_H.jpg",dpi=600,bbox_inches='tight')


# %%
# %% 2、 for rule map of C and Z

# %%  test
feature_1 = 'C'
feature_2 = 'Z'
feature_1_range = C_range
feature_2_range = Z_range

rule = 'Z <= 35 and C <= 127.1 and C > 30'
matches_raw = re.findall(r'(C|Z)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)


# %%
feature_1 = 'C'
feature_2 = 'Z'
feature_1_range = C_range
feature_2_range = Z_range

rule_C_Z['Concentration (mg/L)'] = ''
rule_C_Z['Zeta potential (mV)'] = ''

for i,rule in enumerate(rule_C_Z['rule']):
    matches_raw = re.findall(r'(C|Z)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule) # 
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_C_Z.loc[i,['Concentration (mg/L)','Zeta potential (mV)']] = rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)

rule_C_Z['Concentration (mg/L)'].astype(str)
rule_C_Z['Zeta potential (mV)'].astype(str)
rule_C_Z


# %%
print(rule_C_Z.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

# %%
rule_C_Z_sort_C = {'[25, 37.5]':0,'[25, 75.0]':1,'[25, 150.0]':2,
                 '(37.5, 75.0]':3, '(37.5, 150.0]':4, '(37.5, 200]':5, 
                 '(75.0, 150.0]':6,'(150.0, 200]':7,
       }

# %% simplify the interval for better visualization
print(rule_C_Z.sort_values('Zeta potential (mV)')['Zeta potential (mV)'].unique())

# %%
C_Z_interval_simply_Z = pd.DataFrame({'Raw interval':['[-32.77, -25.835]', '[-32.77, -11.6]', 
                                '[-32.77, -13.03]', '[-32.77, -9.325]',  '[-32.77, -4.945]',
                                '(-13.03, -9.325]', '(-13.03, -4.945]' ,'(-13.03, 44.07]',
                                '(-11.6, 44.07]','(26.17, 44.07]', '(35.27, 44.07]',


                                     ],})
C_Z_interval_simply_Z.loc[0:,'Simply interval'] = simple_interval_func(C_Z_interval_simply_Z['Raw interval'].values[0:],
                                                                       threshold=(Z_range[1]-Z_range[0])*0.05)

len(C_Z_interval_simply_Z['Simply interval'].unique())

# %%
C_Z_interval_simply_Z.to_excel('C_Z_interval_simply_Z.xlsx')
C_Z_interval_simply_Z

# %% replace the interval for better visualization
rule_C_Z_simply = rule_C_Z.copy()
for i,item in enumerate(rule_C_Z_simply['Zeta potential (mV)']):
     index = list(C_Z_interval_simply_Z['Raw interval']).index(item)
     rule_C_Z_simply.loc[i,'Zeta potential (mV)'] = C_Z_interval_simply_Z.loc[index,'Simply interval']
rule_C_Z_simply

# %%
print(len(rule_C_Z_simply['Zeta potential (mV)'].unique()))
rule_C_Z_sort_Z = C_Z_interval_simply_Z['Simply interval'].unique()
rule_C_Z_sort_Z

# %%
rule_map_C_Z_importance = pd.DataFrame(0,index=list(rule_C_Z_sort_C.keys()),columns=list(rule_C_Z_sort_Z),)
rule_map_C_Z_frequency = pd.DataFrame(0,index=list(rule_C_Z_sort_C.keys()),columns=list(rule_C_Z_sort_Z),)
rule_map_C_Z_coefficient = pd.DataFrame(0,index=list(rule_C_Z_sort_C.keys()),columns=list(rule_C_Z_sort_Z),)
rule_map_C_Z_importance

# %%
for i in range(rule_C_Z_simply.shape[0]):
    rule_map_C_Z_importance.loc[rule_C_Z_simply.loc[i,'Concentration (mg/L)'],rule_C_Z_simply.loc[i,'Zeta potential (mV)']] += rule_C_Z.loc[i,'importance']/10
    rule_map_C_Z_coefficient.loc[rule_C_Z_simply.loc[i,'Concentration (mg/L)'],rule_C_Z_simply.loc[i,'Zeta potential (mV)']] += rule_C_Z.loc[i,'coef']/10
    rule_map_C_Z_frequency.loc[rule_C_Z_simply.loc[i,'Concentration (mg/L)'],rule_C_Z_simply.loc[i,'Zeta potential (mV)']] += 1/10

rule_map_C_Z_importance

# %%
rule_map_C_Z_plot = pd.DataFrame(columns=['Zeta potential (mV)','Concentration (mg/L)','Importance','Coefficient','Frequency'])
for C_item in list(rule_C_Z_sort_C.keys()):
    for Z_item in  list(rule_C_Z_sort_Z):
        temp_rule_map = pd.DataFrame({'Zeta potential (mV)':Z_item,
                                        'Concentration (mg/L)':C_item,
                                        'Importance': rule_map_C_Z_importance.loc[C_item,Z_item],
                                        'Coefficient':rule_map_C_Z_coefficient.loc[C_item,Z_item],
                                        'Frequency':rule_map_C_Z_frequency.loc[C_item,Z_item],
                                        }, index=[0])
        rule_map_C_Z_plot = pd.concat([rule_map_C_Z_plot,temp_rule_map],ignore_index=True)
rule_map_C_Z_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (3.5, 3))

palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_C_Z_coefficient.min().min()), abs(rule_map_C_Z_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_C_Z_coefficient.min().min()), abs(rule_map_C_Z_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_C_Z_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_C_Z_importance.max().max()
    print('Warning: get larger importance limit for plot!')

g = sns.scatterplot(
    data=rule_map_C_Z_plot,x="Concentration (mg/L)", y="Zeta potential (mV)", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(50,250*rule_map_C_Z_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')
plt.xlim(-0.5,7.5)

plt.legend(bbox_to_anchor=(2, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

plt.title('RDW')

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

fig.savefig("./Image/Rule_Map_RDW_C_Z.jpg",dpi=600,bbox_inches='tight')





# %% 3、 for rule map of C and T

# %%  test
feature_1 = 'C'
feature_2 = 'T'
feature_1_range = C_range
feature_2_range = T_range

rule = 'C <= 35 and T <= 127.1 and T > 30'
matches_raw = re.findall(r'(C|T)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)


# %%
feature_1 = 'C'
feature_2 = 'T'
feature_1_range = C_range
feature_2_range = T_range

rule_C_T['Concentration (mg/L)'] = ''
rule_C_T['TEM size (nm)'] = ''

for i,rule in enumerate(rule_C_T['rule']):
    matches_raw = re.findall(r'(C|T)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule) # 
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_C_T.loc[i,['Concentration (mg/L)','TEM size (nm)']] = rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)

rule_C_T['Concentration (mg/L)'].astype(str)
rule_C_T['TEM size (nm)'].astype(str)
rule_C_T


# %%
print(rule_C_T.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

# %%
rule_C_T_sort_C = {'[25, 37.5]':0,'[25, 150.0]':1, '(37.5, 200]':2,
                   '(75.0, 200]':3,'(150.0, 200]':4,
       }

# %% simplify the interval for better visualization
print(rule_C_T.sort_values('TEM size (nm)')['TEM size (nm)'].unique())

# %%
C_T_interval_simply_T = pd.DataFrame({'Raw interval':['[12.97, 13.07]','[12.97, 19.155]' ,'[12.97, 38.345]',
                                 '[12.97, 73.455]','[12.97, 113.04]' ,'(13.07, 38.345]', '(13.07, 132.11]' ,
                                 '(14.68, 21.72]', '(19.155, 28.685]', '(28.685, 38.345]' ,
                                 '(38.345, 73.455]','(38.345, 113.04]','(38.345, 132.11]' ,
                                 '(113.04, 132.11]',

                                     ],})
C_T_interval_simply_T.loc[0:,'Simply interval'] = simple_interval_func(C_T_interval_simply_T['Raw interval'].values[0:],
                                                                       threshold=(T_range[1]-T_range[0])*0.05)

len(C_T_interval_simply_T['Simply interval'].unique())

# %%
C_T_interval_simply_T.to_excel('C_T_interval_simply_T.xlsx')
C_T_interval_simply_T

# %% replace the interval for better visualization
rule_C_T_simply = rule_C_T.copy()
for i,item in enumerate(rule_C_T_simply['TEM size (nm)']):
     index = list(C_T_interval_simply_T['Raw interval']).index(item)
     rule_C_T_simply.loc[i,'TEM size (nm)'] = C_T_interval_simply_T.loc[index,'Simply interval']
rule_C_T_simply

# %%
print(len(rule_C_T_simply['TEM size (nm)'].unique()))
rule_C_T_sort_T = C_T_interval_simply_T['Simply interval'].unique()
rule_C_T_sort_T

# %%
rule_map_C_T_importance = pd.DataFrame(0,index=list(rule_C_T_sort_C.keys()),columns=list(rule_C_T_sort_T),)
rule_map_C_T_frequency = pd.DataFrame(0,index=list(rule_C_T_sort_C.keys()),columns=list(rule_C_T_sort_T),)
rule_map_C_T_coefficient = pd.DataFrame(0,index=list(rule_C_T_sort_C.keys()),columns=list(rule_C_T_sort_T),)
rule_map_C_T_importance

# %%
for i in range(rule_C_T_simply.shape[0]):
    rule_map_C_T_importance.loc[rule_C_T_simply.loc[i,'Concentration (mg/L)'],rule_C_T_simply.loc[i,'TEM size (nm)']] += rule_C_T.loc[i,'importance']/10
    rule_map_C_T_coefficient.loc[rule_C_T_simply.loc[i,'Concentration (mg/L)'],rule_C_T_simply.loc[i,'TEM size (nm)']] += rule_C_T.loc[i,'coef']/10
    rule_map_C_T_frequency.loc[rule_C_T_simply.loc[i,'Concentration (mg/L)'],rule_C_T_simply.loc[i,'TEM size (nm)']] += 1/10

rule_map_C_T_importance

# %%
rule_map_C_T_plot = pd.DataFrame(columns=['TEM size (nm)','Concentration (mg/L)','Importance','Coefficient','Frequency'])
for C_item in list(rule_C_T_sort_C.keys()):
    for T_item in  list(rule_C_T_sort_T):
        temp_rule_map = pd.DataFrame({'TEM size (nm)':T_item,
                                        'Concentration (mg/L)':C_item,
                                        'Importance': rule_map_C_T_importance.loc[C_item,T_item],
                                        'Coefficient':rule_map_C_T_coefficient.loc[C_item,T_item],
                                        'Frequency':rule_map_C_T_frequency.loc[C_item,T_item],
                                        }, index=[0])
        rule_map_C_T_plot = pd.concat([rule_map_C_T_plot,temp_rule_map],ignore_index=True)
rule_map_C_T_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (2, 4))

palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_C_T_coefficient.min().min()), abs(rule_map_C_T_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_C_T_coefficient.min().min()), abs(rule_map_C_T_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_C_T_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_C_T_importance.max().max()
    print('Warning: get larger importance limit for plot!')

g = sns.scatterplot(
    data=rule_map_C_T_plot,x="Concentration (mg/L)", y="TEM size (nm)", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(50,250*rule_map_C_T_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')
plt.xlim(-0.5,4.5)

plt.legend(bbox_to_anchor=(2.8, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

norm = plt.Normalize(-larger_coeff_limit, larger_coeff_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.7, anchor=(0, 1.0))

plt.title('RDW')

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/Rule_Map_RDW_C_T.jpg",dpi=600,bbox_inches='tight')





# %% 4、 for rule map of C and B

# %%  test
feature_1 = 'B'
feature_2 = 'C'
feature_1_range = B_range
feature_2_range = C_range

rule = 'C <= 75.0 and C > 37.5 and B <= 27.46 and B > 4.39'
matches_raw = re.findall(r'(C|B)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)


# %%
feature_1 = 'B'
feature_2 = 'C'
feature_1_range = B_range
feature_2_range = C_range

rule_C_B['Concentration (mg/L)'] = ''
rule_C_B['BET surface area (m2/g)'] = ''

for i,rule in enumerate(rule_C_B['rule']):
    matches_raw = re.findall(r'(C|B)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule) # 
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_C_B.loc[i,['BET surface area (m2/g)','Concentration (mg/L)']] = rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)

rule_C_B['Concentration (mg/L)'].astype(str)
rule_C_B['BET surface area (m2/g)'].astype(str)
rule_C_B


# %%
print(rule_C_B.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

# %%
rule_C_B_sort_C = {'[25, 75.0]':0,'[25, 150.0]':1,
                 '(37.5, 75.0]':2,  '(37.5, 200]':3, 
                 '(75.0, 150.0]':4,'(75.0, 200]':5,
                 '(150.0, 200]':6,
       }

# %% simplify the interval for better visualization
print(rule_C_B.sort_values('BET surface area (m2/g)')['BET surface area (m2/g)'].unique())

# %%
C_B_interval_simply_B = pd.DataFrame({'Raw interval':[ '[4.07, 4.39]','[4.07, 27.46]','[4.07, 45.705]',
                                    '(4.39, 27.46]', '(4.39, 45.705]','(27.46, 45.705]','(27.46, 181.475]' ,
                                    '(45.705, 107.75]' ,'(45.705, 200.84]' ,'(73.47, 149.7]',
                                    '(127.18999, 200.84]' ,'(181.475, 200.84]',

                                     ],})
C_B_interval_simply_B.loc[0:,'Simply interval'] = simple_interval_func(C_B_interval_simply_B['Raw interval'].values[0:],
                                                                       threshold=(B_range[1]-B_range[0])*0.05)

len(C_B_interval_simply_B['Simply interval'].unique())

# %%
C_B_interval_simply_B.to_excel('C_B_interval_simply_B.xlsx')
C_B_interval_simply_B

# %% replace the interval for better visualization
rule_C_B_simply = rule_C_B.copy()
for i,item in enumerate(rule_C_B_simply['BET surface area (m2/g)']):
     index = list(C_B_interval_simply_B['Raw interval']).index(item)
     rule_C_B_simply.loc[i,'BET surface area (m2/g)'] = C_B_interval_simply_B.loc[index,'Simply interval']
rule_C_B_simply

# %%
print(len(rule_C_B_simply['BET surface area (m2/g)'].unique()))
rule_C_B_sort_B = C_B_interval_simply_B['Simply interval'].unique()
rule_C_B_sort_B

# %%
rule_map_C_B_importance = pd.DataFrame(0,index=list(rule_C_B_sort_C.keys()),columns=list(rule_C_B_sort_B),)
rule_map_C_B_frequency = pd.DataFrame(0,index=list(rule_C_B_sort_C.keys()),columns=list(rule_C_B_sort_B),)
rule_map_C_B_coefficient = pd.DataFrame(0,index=list(rule_C_B_sort_C.keys()),columns=list(rule_C_B_sort_B),)
rule_map_C_B_importance

# %%
for i in range(rule_C_B_simply.shape[0]):
    rule_map_C_B_importance.loc[rule_C_B_simply.loc[i,'Concentration (mg/L)'],rule_C_B_simply.loc[i,'BET surface area (m2/g)']] += rule_C_B.loc[i,'importance']/10
    rule_map_C_B_coefficient.loc[rule_C_B_simply.loc[i,'Concentration (mg/L)'],rule_C_B_simply.loc[i,'BET surface area (m2/g)']] += rule_C_B.loc[i,'coef']/10
    rule_map_C_B_frequency.loc[rule_C_B_simply.loc[i,'Concentration (mg/L)'],rule_C_B_simply.loc[i,'BET surface area (m2/g)']] += 1/10

rule_map_C_B_importance

# %%
rule_map_C_B_plot = pd.DataFrame(columns=['BET surface area (m2/g)','Concentration (mg/L)','Importance','Coefficient','Frequency'])
for C_item in list(rule_C_B_sort_C.keys()):
    for B_item in  list(rule_C_B_sort_B):
        temp_rule_map = pd.DataFrame({'BET surface area (m2/g)':B_item,
                                        'Concentration (mg/L)':C_item,
                                        'Importance': rule_map_C_B_importance.loc[C_item,B_item],
                                        'Coefficient':rule_map_C_B_coefficient.loc[C_item,B_item],
                                        'Frequency':rule_map_C_B_frequency.loc[C_item,B_item],
                                        }, index=[0])
        rule_map_C_B_plot = pd.concat([rule_map_C_B_plot,temp_rule_map],ignore_index=True)
rule_map_C_B_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (3, 3.5))

palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_C_B_coefficient.min().min()), abs(rule_map_C_B_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_C_B_coefficient.min().min()), abs(rule_map_C_B_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_C_B_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_C_B_importance.max().max()
    print('Warning: get larger importance limit for plot!')

g = sns.scatterplot(
    data=rule_map_C_B_plot,x="Concentration (mg/L)", y="BET surface area (m2/g)", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(50,250*rule_map_C_B_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')
plt.xlim(-0.5,6.5)

plt.legend(bbox_to_anchor=(2.5, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

plt.title('RDW')

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

fig.savefig("./Image/Rule_Map_RDW_C_B.jpg",dpi=600,bbox_inches='tight')

# %%