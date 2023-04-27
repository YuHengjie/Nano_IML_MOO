# %%
from sdmetrics import load_demo
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.utils import get_column_plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %%
raw_data = pd.read_excel("./Raw_data_for_generation.xlsx",index_col = 0,)
generated_data = pd.read_excel("./Generated_data.xlsx",index_col = 0,)
generated_data


# %%
raw_data.insert(0,'No',raw_data.index)
raw_data

# %%
generated_data.insert(0,'No',generated_data.index)
generated_data


# %%
metadata = {
    "fields": {
        'No': {
            'type': 'id', 
            'subtype': 'integer'
        },

        "Composition": {
            "type": "categorical"
        },

        "Concentration (mg/L)": {
            "type": "numerical", 
            "subtype": "integer"
        },

        "TEM size (nm)": {
            "type": "numerical",
            "subtype": "float"
        },

        "Morphology": {
            "type": "categorical"
        },

        "Hydrodynamic diameter (nm)": {
            "type": "numerical",
            "subtype": "float"
        },

        "Zeta potential (mV)": {
            "type": "numerical",
            "subtype": "float"
        },

        "BET surface area (m2/g)": {
            "type": "numerical",
            "subtype": "float"
        },
    },
    'constraints': [],
    'model_kwargs': {},
    'name': None,
    'primary_key': 'No',
    'sequence_index': None,
    'entity_columns': [],
    'context_columns': []
}

# %%
quality_report = QualityReport()
quality_report.generate(raw_data, generated_data, metadata)

# %%
quality_report.get_visualization(property_name='Column Shapes')

# %%
quality_report.get_visualization(property_name='Column Pair Trends')

# %%
column_shape_quality = quality_report.get_details(property_name = 'Column Shapes')
column_shape_quality

# %%
fig = plt.figure(figsize=(4, 4.5))

plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5

sns.barplot(data=column_shape_quality,x='Column',y='Quality Score',dodge=False,palette='flare')

plt.xlabel('')
plt.ylabel('Quality score')
plt.xticks(rotation=45,ha='right')
plt.ylim(0,1)
plt.title('Average shape score = '+str(round(column_shape_quality['Quality Score'].mean(),2)))

fig.savefig("./Image/DG_SHAP_score.jpg",dpi=600,bbox_inches='tight')

# %%
column_pair_trend_quality = quality_report.get_details(property_name = 'Column Pair Trends')
column_pair_trend_quality

# %%
pd_pair_trend = pd.DataFrame(columns=generated_data.columns[1:], index=generated_data.columns[1:],)
pd_pair_trend

# %%
def find_strings(row,feature1,feature2):
    return ((feature1 in row.values) and (feature2 in row.values)) or ((feature2 in row.values) and (feature1 in row.values))

# %%
for index in pd_pair_trend.index:
    for col in pd_pair_trend.columns:
        if index == col:
            pd_pair_trend.loc[index,col] = 1
        else:
            bool_array = column_pair_trend_quality.apply(lambda row: find_strings(row,index,col), axis=1)
            pd_pair_trend.loc[index,col] = column_pair_trend_quality[bool_array]['Quality Score'].values[0]

pd_pair_trend = pd_pair_trend.astype(float)    
pd_pair_trend

# %% 

fig, ax= plt.subplots(figsize = (8, 8))
plt.style.use('default')

h=sns.heatmap(pd_pair_trend, cmap='flare', square=True, center=0.7,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':12},)

bottom, top = ax.get_ylim()

plt.title('Average pair trend score = '+str(round(column_pair_trend_quality['Quality Score'].mean(),2)),fontsize=17)

cb = h.figure.colorbar(h.collections[0],shrink=0.85,) #显示colorbar

cb.ax.tick_params(labelsize=15)  # 设置colorbar刻度字体大小。
cb.set_label('Quality score',fontsize=16)
#ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right',fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=15, rotation_mode='anchor')

fig.savefig("./Image/DG_Pair_trent_score.jpg",dpi=600,bbox_inches='tight')

# %%
# %% 
X = raw_data.drop(columns=['No','Composition','Morphology']) # 'TEM size (nm)'
X_corr = X.copy()
corr = X_corr.corr()

fig, ax= plt.subplots(figsize = (5, 5))
plt.style.use('default')

h=sns.heatmap(corr, cmap='Blues',  square=True, center=0.5,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':10})
bottom, top = ax.get_ylim()
cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=12)  # 设置colorbar刻度字体大小。
cb.set_label('Pearson correlation coefficient',fontsize=14)

#ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right',fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=12, rotation_mode='anchor')

plt.title('Raw data',fontsize=14)

fig.savefig("./Image/DG_raw_corr.jpg",dpi=600,bbox_inches='tight')

# %%
# %% 
X = generated_data.drop(columns=['No','Composition','Morphology']) # 'TEM size (nm)'
X_corr = X.copy()
corr = X_corr.corr()

fig, ax= plt.subplots(figsize = (5, 5))
plt.style.use('default')

h=sns.heatmap(corr, cmap='Blues',  square=True, center=0.5,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':10})
bottom, top = ax.get_ylim()
cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=12)  # 设置colorbar刻度字体大小。
cb.set_label('Pearson correlation coefficient',fontsize=14)

#ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right',fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=12, rotation_mode='anchor')

plt.title('Generated data',fontsize=14)

fig.savefig("./Image/DG_generated_corr.jpg",dpi=600,bbox_inches='tight')


# %%
raw_data = raw_data.drop(columns=['No'])
raw_data

# %%
plt.figure(figsize=(8,10))
plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

for i in range(1,8,1):
    plt.subplot(4,3,i)
    if i in [1,4]:
        if i == 1:
            raw_data[raw_data.columns[i-1]].value_counts().plot(kind='bar',color="#FF8C00",
                edgecolor="black", alpha=0.7, width=1)
            plt.xticks(rotation = 45,horizontalalignment='right')
        elif i == 4:
            raw_data[raw_data.columns[i-1]].value_counts().plot(kind='bar',color="#FF8C00",
                edgecolor="black", alpha=0.7, width=0.15)
            plt.xticks(rotation = 0,horizontalalignment='center')
    else:
        plt.hist(raw_data.iloc[:,i-1], facecolor="#FF8C00", edgecolor="black", alpha=0.7)
    plt.xlabel(raw_data.columns[i-1])
    plt.ylabel("Freqency")
    
plt.tight_layout()
plt.savefig("./Image/dataset_visual_raw.jpg",dpi=600,bbox_inches='tight')

# %%
generated_data = generated_data.drop(columns=['No'])
generated_data

# %%
plt.figure(figsize=(8,10))
plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

for i in range(1,8,1):
    plt.subplot(4,3,i)
    if i in [1,4]:
        if i == 1:
            generated_data[generated_data.columns[i-1]].value_counts().plot(kind='bar',color="#FF8C00",
                edgecolor="black", alpha=0.7, width=1)
            plt.xticks(rotation = 45,horizontalalignment='right')
        elif i == 4:
            generated_data[generated_data.columns[i-1]].value_counts().plot(kind='bar',color="#FF8C00",
                edgecolor="black", alpha=0.7, width=0.15)
            plt.xticks(rotation = 0,horizontalalignment='center')
    else:
        plt.hist(generated_data.iloc[:,i-1], facecolor="#FF8C00", edgecolor="black", alpha=0.7)
    plt.xlabel(generated_data.columns[i-1])
    plt.ylabel("Freqency")
    
plt.tight_layout()
plt.savefig("./Image/dataset_visual_generated.jpg",dpi=600,bbox_inches='tight')

# %%
