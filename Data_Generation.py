# %%
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

import pandas as pd
import numpy as np
import torch

# %% 
data_RDW = pd.read_excel("./Dataset_RDW.xlsx",index_col = 0,)
data_RDW

# %%
data_RDW.columns

# %%
data_RMC = pd.read_excel("./Dataset_RMC.xlsx",index_col = 0,)
data_RMC = data_RMC.drop(columns=['Seedling part'])
data_RMC

# %%
data_RMC.columns

# %%
raw_data = pd.concat([data_RDW.iloc[:,:-1],data_RMC.iloc[:,:-1]])
raw_data = raw_data.drop(columns=['Solubility'])
raw_data

# %%
raw_data = raw_data.sample(frac=1,random_state=1).reset_index(drop=True)
raw_data.dtypes

# %%
raw_data.to_excel("Raw_data_for_generation.xlsx")


# %%
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=raw_data)
metadata.update_column(
    column_name='Concentration (mg/L)',
    sdtype='categorical',
)
metadata

# %%
torch.manual_seed(42)
np.random.seed(42)
synthesizer = GaussianCopulaSynthesizer(metadata,numerical_distributions = {
                    'BET surface area (m2/g)': 'truncnorm',
                    'TEM size (nm)': 'gaussian_kde'},
                    default_distribution = 'gaussian_kde')
synthesizer.fit(raw_data)
generated_data = synthesizer.sample(num_rows=8334)
generated_data = generated_data.drop_duplicates()
generated_data

# %%
generated_data.to_excel('Generated_data.xlsx') 

# %%
generated_data.dtypes

# %%
generated_data['Composition'].value_counts()


# %%
combined_data = pd.concat([raw_data,generated_data], ignore_index=True)
combined_data

# %%
encoded_data = pd.get_dummies(combined_data, columns=['Composition'])
raw_data_encoded = encoded_data[:len(raw_data)]
raw_data_encoded.to_excel('Raw_data_encoded.xlsx')
raw_data_encoded

# %%
generated_data_encoded = encoded_data[len(raw_data):]
generated_data_encoded = generated_data_encoded.reset_index(drop=True)
generated_data_encoded.to_excel('Generated_data_encoded.xlsx')
generated_data_encoded

# %%