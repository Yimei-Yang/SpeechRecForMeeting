with open('./processed-data/features-10M.pkl', 'rb') as f:
  features = pickle.load(f)
nf = features.shape
with open('./processed-data/labels-10M.pkl', 'rb') as f:
  labels = pickle.load(f)
nl = len(labels)

print( "Feature size: {}".format(nf))
print( "Label size: {}".format(nl))
print( "Number of Interruptions: {}".format(np.sum(labels)))

with open('./processed-data/dialogue-acts-whole.pkl', 'rb') as f:
  df_diag_acts = pickle.load(f)
df_diag_acts = addDAoIVariable(df_diag_acts)

df_diag_acts = addDAoIVariable(df_diag_acts)
print("Shape: {}".format(df_diag_acts.shape))
df = df_diag_acts[df_diag_acts.DAoI]
print(df.shape)
print(df.head())
interp_counts = df.value_counts(subset= ['participant', 'DAoI'])

import seaborn as sns
# from matplotlib.pyplot import plt
g = sns.histplot(x = interp_counts, bins = 60)
g.set_xlabel("Interruption count")
g.set_xlabel("Frequency")
g.set_title("Interruption count by participants")

# sns.histplot(x = df['participant'])
# print(df['participant'].value_counts())