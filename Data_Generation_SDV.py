# %%
from sdv.tabular import TVAE
import pandas as pd
import numpy as np
import torch

# %% 
data_RDW = pd.read_excel("./dataset_RDW.xlsx",index_col = 0,)
data_RDW.head()

# %%
data_RDW.columns

# %%
data_RMC = pd.read_excel("./dataset_RMC.xlsx",index_col = 0,)
data_RMC.head()

# %%
data_RMC.columns

# %%
Raw_data = pd.concat([data_RDW.iloc[:,:-1],data_RMC.iloc[:,:-1]])
Raw_data.shape


# %%
Raw_data = Raw_data.sample(frac=1,random_state=1).reset_index(drop=True)
Raw_data['Concentration (mg/L)'] = Raw_data['Concentration (mg/L)'].astype(object)
Raw_data.dtypes

# %%
Raw_data.to_excel("Raw_data_for_generation.xlsx")

# %%
torch.manual_seed(42)
np.random.seed(42)
model = TVAE()
model.fit(Raw_data)

# %%
generated_data = model.sample(num_rows=8334)
generated_data

# %%
generated_data.to_excel('Generated_data.xlsx') 

# %%
generated_data.dtypes

# %%
composition_encoding = {'CuO':1, 'α-Fe2O3':2, 'TiO2':3, 'ZnO':4, 'Fe3O4':5, 'SiO2':6, 'CeO2':7,'γ-Fe2O3':8}
morphology_encoding  = {'Compound':1, 'Spherical':2}

encoded_Raw_data = pd.get_dummies(Raw_data, columns=['Composition', 'Morphology'], dummy_na=False)
encoded_Raw_data = encoded_Raw_data.replace({'Composition_CuO': composition_encoding,
                                 'Composition_α-Fe2O3': composition_encoding,
                                 'Composition_TiO2': composition_encoding,
                                 'Composition_ZnO': composition_encoding,
                                 'Composition_Fe3O4': composition_encoding,
                                 'Composition_SiO2': composition_encoding,
                                 'Composition_CeO2': composition_encoding,
                                 'Composition_γ-Fe2O3': composition_encoding,
                                 'Morphology_Compound': morphology_encoding,
                                 'Morphology_Spherical': morphology_encoding})

encoded_Raw_data

# %%

encoded_Generated_data = pd.get_dummies(generated_data, columns=['Composition', 'Morphology'], dummy_na=False)
encoded_Generated_data = encoded_Generated_data.replace({'Composition_CuO': composition_encoding,
                                 'Composition_α-Fe2O3': composition_encoding,
                                 'Composition_TiO2': composition_encoding,
                                 'Composition_ZnO': composition_encoding,
                                 'Composition_Fe3O4': composition_encoding,
                                 'Composition_SiO2': composition_encoding,
                                 'Composition_CeO2': composition_encoding,
                                 'Composition_γ-Fe2O3': composition_encoding,
                                 'Morphology_Compound': morphology_encoding,
                                 'Morphology_Spherical': morphology_encoding})

encoded_Generated_data

# %%
encoded_Raw_data.to_excel('Raw_data_encoded.xlsx')
encoded_Generated_data.to_excel('Generated_data_encoded.xlsx')

# %%
