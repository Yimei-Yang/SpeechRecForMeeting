#%%
import pickle, os
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import *
from collections import Counter
# with open('./processed-data/features-10M.pkl', 'rb') as f:
#   features = pickle.load(f)
# nf = features.shape
# with open('./processed-data/labels-10M.pkl', 'rb') as f:
#   labels = pickle.load(f)
# nl = len(labels)

# print( "Feature size: {}".format(nf))
# print( "Label size: {}".format(nl))
# print( "Number of Interruptions: {}".format(np.sum(labels)))
print(os.chdir("/Users/vivcrowe/Library/CloudStorage/OneDrive-ConcordiaUniversity-Canada/AI4Good/"))
print(os.getcwd())

with open('processed-data/dialogue-acts-prepped.pkl', 'rb') as f:
  df_diag_acts = pickle.load(f)
df_diag_acts.head()

# %%

df_count = pd.DataFrame(df_diag_acts.groupby(['participant'])['participant'].count())
df_count['Gender']  = [j[0] for j in df_count.index]
df_count.rename(columns = {'participant': 'count'}, inplace=True )
df_count.head()

# for p in df_diag_acts['participant'].unique():
#   counts_data[p] = df_diag_acts[df_diag_acts['participant'] == p]

# %%
from matplotlib import colors as mcolors
sns.set_style("ticks")

colors_dict = {'fs_blue': "#283C63", 'fs_red': '#FF7069','fs_beige':'#FF7069', 'fs_grey': '#FF7069' }
palette = {'f': colors_dict['fs_red'], 'm':  colors_dict['fs_blue']}

g = sns.histplot(x = df_count['count'], hue = df_count['Gender'], bins = 30, palette= palette, alpha=0.9)
g.set_ylabel("Number of People", fontsize = 12)
g.set_xlabel("Number of Interruptions", fontsize =12)
g.set_title("Interruption frequency", fontsize = 18)
sns.despine()
g

fig = g.get_figure()
fig.savefig("interp_freq_plot.png") 

df_count.groupby(['Gender']).count()

# %%
from scipy import stats

# stats.ranksums(df_count['count'][df_count['Gender']=='f'], df_count['count'][df_count['Gender']=='m'])


# %%
