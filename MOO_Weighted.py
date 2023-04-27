# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %%
RDW_prob = pd.read_excel("./Predict_RDW.xlsx",index_col = 0,)
RDW_prob

# %%
f1_RDW = RDW_prob['Average']
f1_RDW

# %%
RMC_prob = pd.read_excel("./Predict_RMC.xlsx",index_col = 0,)
RMC_prob

# %%
f2_RMC = RMC_prob['Average']
f2_RMC


# %%
data_generated = pd.read_excel("./Generated_data.xlsx",index_col = 0,)

full_data = data_generated.copy()
full_data['RDW'] = RDW_prob['Average']
full_data['RMC'] = RMC_prob['Average']
full_data

# %%
full_data['Equal-weighted'] = 0.5*RDW_prob['Average']+0.5*(1-RMC_prob['Average'])
full_data['RDW-first'] = 0.7*RDW_prob['Average']+0.3*(1-RMC_prob['Average'])
full_data['RMC-first'] = 0.3*RDW_prob['Average']+0.7*(1-RMC_prob['Average'])
full_data

# %%
color = ['#FFD700','#4CB02C','#B4CD13','#D6D70B','#7EBE20']
for i,object in enumerate(full_data.columns[-5:]):
    plt.figure(figsize=(2,2))
    plt.style.use('default')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    plt.hist(full_data[object],bins=20, facecolor=color[i], edgecolor="black", alpha=0.7)

    plt.xlabel(object)
    if i == 0 or i == 2:
        plt.ylabel("Freqency")
    else:
        plt.ylabel("")

    plt.xlim(-0.05,1.05)
    plt.savefig("./Image/Object_distribution_%s.jpg"%object,dpi=600,bbox_inches='tight')

# %%
Equal_weighted_ten = full_data.sort_values(['Equal-weighted'], ascending=[False]).iloc[0:10,:]
Equal_weighted_ten.to_excel('Equal_weighted_ten.xlsx')
Equal_weighted_ten

# %%
Equal_weighted_38 = full_data.sort_values(['Equal-weighted'], ascending=[False]).iloc[0:38,:]
Equal_weighted_38.to_excel('Equal_weighted_38.xlsx')
Equal_weighted_38

# %%
RDW_first_ten = full_data.sort_values(['RDW-first'], ascending=[False]).iloc[0:10,:]
RDW_first_ten.to_excel('RDW_first_ten.xlsx')
RDW_first_ten

# %%
RMC_first_ten = full_data.sort_values(['RMC-first'], ascending=[False]).iloc[0:10,:]
RMC_first_ten.to_excel('RMC_first_ten.xlsx')
RMC_first_ten

# %%
list1 = list(Equal_weighted_ten.index)
list2 = list(RDW_first_ten.index)
list3 = list(RMC_first_ten.index)

freq_dict = {}

for elem in list1 + list2 + list3:
    if elem in freq_dict:
        freq_dict[elem] += 1
    else:
        freq_dict[elem] = 1

repeated_elems = [elem for elem in freq_dict if freq_dict[elem] == 3]

print(repeated_elems)

# %%
