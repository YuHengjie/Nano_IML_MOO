# %%
import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


# %%
raw_AD = pd.read_excel("Data_std_AD.xlsx",sheet_name=0,index_col=[0],) # Raw dateset
gererated_AD = pd.read_excel("Data_std_AD.xlsx",sheet_name=1,index_col=[0],) # Generated data
raw_AD

# %%
raw_snew_num = raw_AD[raw_AD['Snew']!='Not Applicable'].shape[0]
outlier_num = raw_AD[raw_AD['Outlier Info.']!='-'].shape[0]
print('The number of instances needed to calculate Snew in the raw data:',raw_snew_num)
print('The number of outlier instances in the raw data:',outlier_num)

# %%
Snew_raw = raw_AD[raw_AD['Snew']!='Not Applicable']
Snew_raw

# %%
generated_snew_num = gererated_AD[gererated_AD['Snew']!='Not Applicable'].shape[0]
outlier_num = gererated_AD[gererated_AD['AD Info.']!='-'].shape[0]
print('The number of instances needed to calculate Snew in the generated data:',generated_snew_num)
print('The number of outlier instances in the generated data:',outlier_num)

# %%
Snew_generated = gererated_AD[gererated_AD['Snew']!='Not Applicable']
Snew_generated

# %%
random_No = random.sample(range(1,raw_snew_num+generated_snew_num+1),raw_snew_num+generated_snew_num)
Snew_raw['Random No'] = random_No[0:raw_snew_num]
Snew_generated['Random No'] = random_No[raw_snew_num:raw_snew_num+generated_snew_num]


# %%
fig = plt.figure(figsize=(4, 4))

plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5

plt.scatter(Snew_generated['Random No'], Snew_generated['Snew'], c='#4682B4', s=60, alpha=0.65, label="Generated data",  marker="*")
plt.scatter(Snew_raw['Random No'], Snew_raw['Snew'], c='#FF4500', s=60, alpha=0.65, label="Raw data", marker="h")

# plt.scatter(np.log10(new_test.index), new_test['leverages'], c='black', s=200, label="New instance", marker="p")

# plt.axhline(y = 3, color = 'blue', linestyle = '-', linewidth=3 , label= 'Threshold' )
plt.title("Applicability domain",fontsize=11)
plt.xlabel("Random instance No.",fontsize=11)

plt.ylabel('S'+'${_n}$'+'${_e}$'+'${_w}$',fontsize=11)
plt.legend(loc=1)
plt.ylim(1.65, 2.8)
plt.xlim(-50, raw_snew_num+generated_snew_num+50)
plt.grid()
fig.savefig("./Image/DG_Domain_applicability.jpg",dpi=600,bbox_inches='tight')

# %%
