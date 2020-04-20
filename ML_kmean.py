
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import datetime

# Data Management

df= pd.read_pickle("./file.pkl")
df_sample = df.sample(frac=0.1, random_state=123, axis=None)
df_clean = df_sample.fillna(0)

var_list=['A', 'B', 'C', 'D', 'E']

cluster = df_clean[var_list]

cluster.describe()

print(datetime.datetime.now())

# standardize clustering variables to have mean=0 and sd=1
clustervar=cluster.copy()
for var_name in var_list:
	clustervar[var_name]=preprocessing.scale(clustervar[var_name].astype('float64'))

# split data into train and test sets
# clus_train, clus_test = train_test_split(clustervar, test_size=0.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,20)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clustervar)
    clusassign=model.predict(clustervar)
    meandist.append(sum(np.min(cdist(clustervar, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clustervar.shape[0])
    print(str(datetime.datetime.now()) + "  k=" +str(k)) 
    
"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.axvline(x=7,color='gray', linewidth=1.0, linestyle='--')
plt.savefig('./output/user_reaction_part_variable/1_choose_number_of_cluster', dpi=500)
print(datetime.datetime.now())


################################### Interpret 7 cluster solution ################################### 
model7=KMeans(n_clusters=7)
model7.fit(clustervar)
clusassign=model7.predict(clustervar)

df_clean['group'] = clusassign.tolist()
df_clean.to_pickle("./output/cluster.pkl")
df_clean.to_csv("./output/cluster.csv")

print(datetime.datetime.now())
print(df_clean.count())

df_new = pd.read_csv("./output/cluster.csv")
print (df_new.dtypes)
group_count= df_new.groupby("group")["name"].count()
print (group_count)
writer = pd.ExcelWriter('./output/cluster_count.xlsx')
group_count.to_frame(name='count').to_excel(writer, "name_count")
writer.save()




# plot clusters
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clustervar)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model7.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 7 Clusters')
plt.savefig('./output/2_Scatterplot_of_Canonical_Variables_for_7_Clusters', dpi=500)
print(datetime.datetime.now())
