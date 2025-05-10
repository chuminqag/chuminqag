
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load và chuẩn bị dữ liệu
df = pd.read_csv('results.csv')
# Lấy các cột numeric
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
X = df[numeric_cols].fillna(0)

# 2. Scale dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Tìm k tối ưu với Elbow và Silhouette
Ks = range(2, 11)
inertia = []
sil_scores = []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Vẽ Elbow
plt.figure()
plt.plot(Ks, inertia, '-o')
plt.xlabel('Số cụm k')
plt.ylabel('Tổng Inertia')
plt.title('Elbow Method')
plt.savefig('elbow.png')
plt.close()

# Vẽ Silhouette
plt.figure()
plt.plot(Ks, sil_scores, '-o')
plt.xlabel('Số cụm k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.savefig('silhouette.png')
plt.close()

# Chọn k với silhouette cao nhất
best_k = Ks[int(np.argmax(sil_scores))]
print(f"Best k by silhouette: {best_k}")

# 4. Chạy KMeans với k đã chọn
kmeans = KMeans(n_clusters=best_k, random_state=0).fit(X_scaled)
df['Cluster'] = kmeans.labels_

# 5. PCA xuống 2 chiều và vẽ scatter với màu theo cụm
pca = PCA(n_components=2, random_state=0)
pcs = pca.fit_transform(X_scaled)
df['PC1'], df['PC2'] = pcs[:,0], pcs[:,1]

plt.figure(figsize=(8,6))
for c in range(best_k):
    subset = df[df['Cluster']==c]
    plt.scatter(subset['PC1'], subset['PC2'], label=f'Cụm {c}', alpha=0.6)
plt.legend()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA & KMeans Clusters')
plt.savefig('pca_clusters.png')
plt.close()

# 6. Ghi file comment phân tích
with open('cluster_comments.txt', 'w', encoding='utf-8') as f:
    f.write(f"Silhouette scores for k=2..10: {dict(zip(Ks, sil_scores))}\\n")
    f.write(f"Selected k = {best_k}\\n\\n")
    for i in range(best_k):
        cent = scaler.inverse_transform(kmeans.cluster_centers_[i])
        f.write(f"Cụm {i} centroid:\\n")
        for col, val in zip(numeric_cols, cent):
            f.write(f"  {col}: {val:.2f}\\n")
        f.write("\\n")

