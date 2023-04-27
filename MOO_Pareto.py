# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.manifold import TSNE

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
feasible_df = pd.DataFrame({'f1_RDW': f1_RDW, 'f2_RMC': f2_RMC})
feasible_df = feasible_df.sort_values(['f1_RDW', 'f2_RMC'], ascending=[False, True])
feasible_df

# %%
non_dominated = []

for i, row in feasible_df.iterrows():
    dominated = False

    for p in non_dominated:
        if p['f1_RDW'] >= row['f1_RDW'] and p['f2_RMC'] <= row['f2_RMC']:
            dominated = True
            break
    if not dominated:
        non_dominated.append(row)
pareto_optimal_solution = pd.DataFrame(non_dominated)
pareto_optimal_solution

# %%
fig = plt.figure(figsize=(4, 4))

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.50

plt.scatter(feasible_df['f1_RDW'], feasible_df['f2_RMC'], c='#4682B4', s=30, alpha=0.5,  marker="h",label='Feasible solutions')
plt.scatter(pareto_optimal_solution['f1_RDW'], pareto_optimal_solution['f2_RMC'], c='#FF4500', s=50, 
             marker="*",label='Non-dominated solutions')
plt.plot(pareto_optimal_solution.sort_values(by=['f1_RDW'])['f1_RDW'], pareto_optimal_solution.sort_values(by=['f1_RDW'])['f2_RMC'],
         c='#FF4500', label='Pareto front',)

plt.title("Objective space")
plt.xlabel('RDW')
plt.ylabel('RMC')

plt.legend(loc=2)
plt.ylim(-0.05,1.05)
plt.xlim(-0.05,1.05)
plt.grid()
fig.savefig("./Image/Pareto_optimality_full.jpg",dpi=600,bbox_inches='tight')


# %%
feasible_df = pd.DataFrame({'f1_RDW': f1_RDW, 'f2_RMC': f2_RMC})
feasible_df = feasible_df[feasible_df['f1_RDW']>0.7]
feasible_df = feasible_df.sort_values(['f1_RDW', 'f2_RMC'], ascending=[False, True])
feasible_df

# %%
non_dominated = []

for i, row in feasible_df.iterrows():
    dominated = False

    for p in non_dominated:
        if p['f1_RDW'] >= row['f1_RDW'] and p['f2_RMC'] <= row['f2_RMC']:
            dominated = True
            break
    if not dominated:
        non_dominated.append(row)
pareto_optimal_solution = pd.DataFrame(non_dominated)
pareto_optimal_solution

# %%
fig = plt.figure(figsize=(4, 4))

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.50

plt.scatter(feasible_df['f1_RDW'], feasible_df['f2_RMC'], c='#4682B4', s=30, alpha=0.5,  marker="h",label='Feasible solutions')
plt.scatter(pareto_optimal_solution['f1_RDW'], pareto_optimal_solution['f2_RMC'], c='#FF4500', s=50, 
             marker="*",label='Non-dominated solutions')
plt.plot(pareto_optimal_solution.sort_values(by=['f1_RDW'])['f1_RDW'], pareto_optimal_solution.sort_values(by=['f1_RDW'])['f2_RMC'],
         c='#FF4500', label='Pareto front',)

plt.title("Objective space (RDW > 0.7)")
plt.xlabel('RDW')
plt.ylabel('RMC')

plt.legend(loc=2)
plt.ylim(-0.05,1.05)
plt.xlim(0.68,1.0)
plt.grid()
fig.savefig("./Image/Pareto_optimality.jpg",dpi=600,bbox_inches='tight')

# %%
data_generated = pd.read_excel("./Generated_data.xlsx",index_col = 0,)
full_data = data_generated.copy()
full_data['RDW'] = RDW_prob['Average']
full_data['RMC'] = RMC_prob['Average']
full_data['Distance to ideal point'] = np.sqrt((full_data['RDW'] - 1) ** 2 + full_data['RMC'] ** 2)

full_data

# %%
Pareto_optimal_solution_distance = full_data.loc[pareto_optimal_solution.index,:].sort_values(['Distance to ideal point'], ascending=[True])
Pareto_optimal_solution_distance.to_excel('Pareto_optimal_solution.xlsx')
Pareto_optimal_solution_distance





# %%
data_generated_encoded = pd.read_excel("./Generated_data_encoded.xlsx",index_col = 0,)
data_generated_encoded

# %%
X_embedded = TSNE(n_components=2, learning_rate='auto',
               init='pca', perplexity=30,random_state=42,n_jobs=-1).fit_transform(data_generated_encoded)
X_embedded

# %%
fig = plt.figure(figsize=(4, 4))

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.50

plt.scatter(X_embedded[:,0], X_embedded[:,1], c='#4682B4', s=20, alpha=0.1,  marker="h",label='Feasible solutions')
plt.scatter(X_embedded[Pareto_optimal_solution_distance.index,0], X_embedded[Pareto_optimal_solution_distance.index,1],
             c='#FF4500', s=50, marker="*",label='Non-dominated solutions')

plt.title("Decision space")
plt.xlabel('TSNE dimension 1')
plt.ylabel('TSNE dimension 2')

plt.legend(loc=2)
plt.grid()
fig.savefig("./Image/Pareto_optimality_decision.jpg",dpi=600,bbox_inches='tight')

# %%
