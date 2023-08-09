# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shap
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %%
X_raw = pd.read_excel("./Generated_data.xlsx",index_col = 0,)
X_new = X_raw.drop(columns=['Composition']) 
X_new

# %%
for i in range(1,11,1):

    exec("shap_values = np.load(\"./SHAP_values/SHAP_values_{}.npy\", allow_pickle=True)".format(i))

    if i == 1:
        shap_values_all = shap_values
    else:
        shap_values_all = np.append(shap_values_all,shap_values,axis=0)
shap_values_all = shap_values_all[:,0:5]
print(shap_values_all.shape)

# %%
feature_name_list = list(X_new.columns)
feature_name_list

# %%
pd_shap_values = pd.DataFrame(index=range(0,shap_values_all.shape[0]),columns=feature_name_list)

for i in range(0,len(feature_name_list)):
    feature_shap_values = shap_values_all[:,i]
    feature_shap_values = [feature_shap_values[j].values for j in range(0,feature_shap_values.shape[0])]
    pd_shap_values.iloc[:,i] = feature_shap_values

pd_shap_values

# %%
shap_importance = pd.DataFrame(index = range(0,len(feature_name_list)),columns=['Feature','SHAP mean value'])
shap_importance

# %%
for i in range(0,len(feature_name_list)):
    shap_importance.loc[i,'Feature'] = feature_name_list[i]
    shap_importance.loc[i,'SHAP mean value'] = abs(pd_shap_values[feature_name_list[i]]).mean()
shap_importance = shap_importance.sort_values(by=['SHAP mean value'],ascending=[False])
shap_importance


# %%

color_list = ['#8DA0CB','#65C2A5','#FCFEA4','#FC8D62','#E78AC3']# ['#FB9B06','#ED6825','#A42C60','#4A0B6A']
fig, ax = plt.subplots(figsize=(3.5,2.5)) 
plt.style.use('classic')
plt.bar(shap_importance['Feature'],shap_importance['SHAP mean value'],width=0.4,color=color_list,edgecolor='k',alpha=0.6)

plt.xticks(rotation=45,ha='right')
plt.ylim(0,0.1)
plt.xlim(-0.5,4.5)
plt.ylabel('Average SHAP importance')

fig.savefig("./Image/SHAP_RuleFit_importance_feature_NPI.jpg",dpi=600,bbox_inches='tight')


# %%
pd_shap_values['Model'] = pd_shap_values.index//X_new.shape[0]+1
pd_shap_values

# %%
SHAP_feature_name = feature_name_list.copy()
SHAP_feature_name = ['SHAP_'+ feature_name for feature_name in feature_name_list]
SHAP_feature_name

# %%
pd_shap_values_average = pd.DataFrame(0,index=X_new.index,columns=SHAP_feature_name)
pd_shap_values_average

# %%
for i in range(1,11):
    pd_shap_values_average.iloc[:,:] += pd_shap_values[pd_shap_values['Model']==i].iloc[:,0:5].values/10
pd_shap_values_average

# %%
pd_shap_values_average_sum = pd_shap_values_average.copy()
pd_shap_values_average_sum['Sum'] = pd_shap_values_average_sum.apply(lambda x: x.sum(), axis=1)
pd_shap_values_average_sum

# %%
pd_shap_values_average_sum = pd_shap_values_average_sum.sort_values(by=['Sum'],ascending=[False])
pd_shap_values_average_sum

# %%
SHAP_top_49 = pd_shap_values_average_sum.iloc[0:49,:]
SHAP_top_49

# %%
SHAP_49_index = SHAP_top_49.index
np.save("SHAP_49_index.npy", SHAP_49_index)


# %%
df_plot_all = pd.concat([X_new, pd_shap_values_average], axis=1)
df_plot_all

# %%
'''
random.seed(42)
shap_index = random.sample(range(0,X_new.shape[0]),2000)
df_plot_2000 = df_plot_all.iloc[shap_index,:]
df_plot_2000
'''

# %% global SHAP values for numerical features

palette = ['#8DA0CB','#65C2A5','#FC8D62','#E78AC3']

feature = 'Hydrodynamic diameter (nm)'

fig, ax = plt.subplots(figsize=(4.2, 3.5))
plt.xlabel(feature)
plt.ylabel('Average SHAP value')

sns.scatterplot(data = df_plot_all,x=feature, y='SHAP_'+feature, hue='Concentration (mg/L)',
                palette = sns.color_palette(palette, 4),edgecolor="black", linewidth=0.2)# 

plt.legend(bbox_to_anchor=(1.25, 0.7), loc='upper right', 
        borderaxespad=0,labelspacing=0.8, scatterpoints=1,
        frameon = False, ncol=1,prop = {'size':12}
        )

ax.text(1200, 0.05, 'Concentration (mg/L)', rotation=270, va='center', ha='center')
plt.xticks([200,400,600,800,1000])
plt.ylim([-0.16,0.26])
plt.yticks([-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25])

plt.xlim([160,970])

fig.savefig("./Image/SHAP_values_%s.jpg"%feature[0:6],dpi=600,bbox_inches='tight')

# %%
feature = 'BET surface area (m2/g)'

fig, ax = plt.subplots(figsize=(4.2, 3.5))
plt.xlabel(feature)
plt.ylabel('Average SHAP value')
sns.scatterplot(data = df_plot_all,x=feature, y='SHAP_'+feature, hue='Concentration (mg/L)',
                palette = sns.color_palette(palette, 4),edgecolor="black", linewidth=0.2)# 

plt.legend(bbox_to_anchor=(1.25, 0.7), loc='upper right', 
        borderaxespad=0,labelspacing=0.8, scatterpoints=1,
        frameon = False, ncol=1,prop = {'size':12}
        )

ax.text(270, 0.0375, 'Concentration (mg/L)', rotation=270, va='center', ha='center')
plt.xlim([-5,210])
plt.ylim([-0.17,0.245])
plt.yticks([-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2])

fig.savefig("./Image/SHAP_values_%s.jpg"%feature[0:6],dpi=600,bbox_inches='tight')

# %%
feature = 'Zeta potential (mV)'

fig, ax = plt.subplots(figsize=(4.2, 3.5))
plt.xlabel(feature)
plt.ylabel('Average SHAP value')
sns.scatterplot(data = df_plot_all,x=feature, y='SHAP_'+feature, hue='Concentration (mg/L)',
                palette = sns.color_palette(palette, 4),edgecolor="black", linewidth=0.2)# 

plt.legend(bbox_to_anchor=(1.25, 0.7), loc='upper right', 
        borderaxespad=0,labelspacing=0.8, scatterpoints=1,
        frameon = False, ncol=1,prop = {'size':12}
        )

ax.text(70, 0.045, 'Concentration (mg/L)', rotation=270, va='center', ha='center')

plt.xticks([-40,-20,0,20,40])
plt.yticks([-0.1,-0.05,0,0.05,0.1,0.15,0.2])
plt.ylim([-0.13,0.22])
plt.xlim([-36,47])

fig.savefig("./Image/SHAP_values_%s.jpg"%feature[0:6],dpi=600,bbox_inches='tight')

# %%
feature = 'TEM size (nm)'

fig, ax = plt.subplots(figsize=(4.2, 3.5))
plt.xlabel(feature)
plt.ylabel('Average SHAP value')
sns.scatterplot(data = df_plot_all,x=feature, y='SHAP_'+feature, hue='Concentration (mg/L)',
                palette = sns.color_palette(palette, 4),edgecolor="black", linewidth=0.2)# 

plt.legend(bbox_to_anchor=(1.25, 0.7), loc='upper right', 
        borderaxespad=0,labelspacing=0.8, scatterpoints=1,
        frameon = False, ncol=1,prop = {'size':12}
        )

ax.text(175, -0.0175, 'Concentration (mg/L)', rotation=270, va='center', ha='center')

plt.ylim([-0.125,0.09])
plt.xlim([7,138])

fig.savefig("./Image/SHAP_values_%s.jpg"%feature[0:6],dpi=600,bbox_inches='tight')


# %%
feature = 'Concentration (mg/L)'

fig, ax = plt.subplots(figsize=(4, 2.8))
sns.boxenplot(data=df_plot_all, x=feature, y='SHAP_'+feature,palette = sns.color_palette(palette, 4))
plt.xlabel(feature)
plt.ylabel('Average SHAP value')
fig.savefig("./Image/SHAP_values_%s.jpg"%feature[0:6],dpi=600,bbox_inches='tight')



# %%
pd_shap_values_average.iloc[1809,:]

# %%
X_new.iloc[1809,:]

# %%
plot_1809_df = pd.DataFrame(columns=['Feature','Feature value','Average SHAP value'],index=range(0,len(pd_shap_values_average.columns)))
plot_1809_df['Feature'] = feature_name_list
plot_1809_df['Feature value'] = X_new.iloc[1809,:].values
plot_1809_df['Average SHAP value'] = pd_shap_values_average.iloc[1809,:].values

plot_1809_df['Combination'] = [plot_1809_df.loc[i,'Feature'] + ' = ' + str(plot_1809_df.loc[i,'Feature value']) for i in plot_1809_df.index]
plot_1809_df = plot_1809_df.sort_values(by=['Average SHAP value'],ascending=[True])
plot_1809_df

# %%
fig, ax= plt.subplots(figsize = (5,3))
plt.style.use('seaborn-ticks')
plt.margins(0.05)
plt.grid(linestyle=(0, (1, 6.5)),color='#B0B0B0',zorder=0)
plt.barh(plot_1809_df['Combination'], plot_1809_df['Average SHAP value'],
         edgecolor = "black", zorder=3,color='#F08080',alpha=0.8)
plt.xlabel('Average SHAP value')
plt.title('Local interpretation for instance No.1809')
plt.xlim(-0.04,0.20)

ax.tick_params(top=False,
            bottom=True,
            left=True,
            right=False)

height = plot_1809_df['Average SHAP value'].values
for i in range(0,plot_1809_df.shape[0]):
    if height[i]>0:
        plt.text(height[i]-0.012,i-0.1,"%.2f" %height[i],ha = 'center',color='black',)
    else:
        plt.text(height[i]+0.03,i-0.1,"%.2f" %height[i],ha = 'center',color='black',)

fig.savefig("./Image/SHAP_local_1809.jpg",dpi=600,bbox_inches='tight')







# %%
pd_shap_values_average.iloc[7533,:]

# %%
X_new.iloc[7533,:]

# %%
plot_7533_df = pd.DataFrame(columns=['Feature','Feature value','Average SHAP value'],index=range(0,len(pd_shap_values_average.columns)))
plot_7533_df['Feature'] = feature_name_list
plot_7533_df['Feature value'] = X_new.iloc[7533,:].values
plot_7533_df['Average SHAP value'] = pd_shap_values_average.iloc[7533,:].values

plot_7533_df['Combination'] = [plot_7533_df.loc[i,'Feature'] + ' = ' + str(plot_7533_df.loc[i,'Feature value']) for i in plot_7533_df.index]
plot_7533_df = plot_7533_df.sort_values(by=['Average SHAP value'],ascending=[True])
plot_7533_df

# %%
fig, ax= plt.subplots(figsize = (5,3))
plt.style.use('seaborn-ticks')
plt.margins(0.05)
plt.grid(linestyle=(0, (1, 6.5)),color='#B0B0B0',zorder=0)
plt.barh(plot_7533_df['Combination'], plot_7533_df['Average SHAP value'],
         edgecolor = "black", zorder=3,color='#F08080',alpha=0.8)
plt.xlabel('Average SHAP value')
plt.title('Local interpretation for instance No.7533')
plt.xlim(-0.04,0.20)

ax.tick_params(top=False,
            bottom=True,
            left=True,
            right=False)

height = plot_7533_df['Average SHAP value'].values
for i in range(0,plot_7533_df.shape[0]):
    if height[i]>0:
        plt.text(height[i]-0.012,i-0.1,"%.2f" %height[i],ha = 'center',color='black',)
    else:
        plt.text(height[i]+0.015,i-0.1,"%.2f" %height[i],ha = 'center',color='black',)

fig.savefig("./Image/SHAP_local_7533.jpg",dpi=600,bbox_inches='tight')

# %%
