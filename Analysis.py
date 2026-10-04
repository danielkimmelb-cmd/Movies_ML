import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

credits_df = pd.read_csv('Datasets/credits_with_wiki.csv')
titles_df = pd.read_csv('Datasets/titles.csv')
merged_df = pd.merge(titles_df, credits_df, on = 'id', how ='inner')
merged_df['isFamous'] = (merged_df['page_size'] > 0).astype(int)

selected_features = ['isFamous','tmdb_popularity']

data = merged_df[selected_features].dropna() 
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

num_clusters = 4

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(data_scaled)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 6))
for cluster_label in range(num_clusters):
    plt.scatter(data_pca[data['cluster'] == cluster_label, 0],
                data_pca[data['cluster'] == cluster_label, 1],
                label=f'Cluster {cluster_label}')
plt.title('Clustering of Movies Based on Actor Fame')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
