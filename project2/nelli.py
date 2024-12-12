from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_rf_top3)

# Heatmap of confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Stroke'], yticklabels=['Healthy', 'Stroke'])
plt.title('Confusion Matrix for Random Forest (Top 3 Features)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

results = {
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "Random Forest": accuracy_score(y_test, y_pred_rf),
    "SVM": accuracy_score(y_test, y_pred_svm)
}
print("Accuracy:\n", results,'\n')
print("Error Rate:\n", {k: 1-v for k, v in results.items()})

# Accuracy/Error Rate Bar Plot
models = list(results.keys())
accuracy = list(results.values())
error_rate = [1 - acc for acc in accuracy]

plt.bar(models, accuracy, color='green', alpha=0.6,label='Accuracy')
plt.bar(models, error_rate, color='yellow', alpha=0.6, label='Error Rate',bottom=accuracy)
plt.legend()
plt.title("Accuracy and Error Rate Comparison")
plt.show()

ax.set_xlabel('Age')
ax.set_ylabel('Average Glucose Level')
ax.set_zlabel('BMI')
ax.set_title('3D Scatter Plot - Logistic Regression')

plt.legend(loc='upper right')
plt.show()

X_unsupervised = df.drop(columns=['stroke'])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_unsupervised)
kmeans_labels = kmeans.labels_
kmeans_score = silhouette_score(X_unsupervised, kmeans_labels)

# Print Silhouette Score
print("K-Means Silhouette Score:", kmeans_score)

plt.scatter(X_unsupervised.iloc[:, 0], X_unsupervised.iloc[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering Scatter Plot')
plt.show()

X = df[['age', 'avg_glucose_level']]
# Նորմալացնում ենք տվյալները
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans մոդել
kmeans = KMeans(n_clusters=3, random_state=42)  # Կլաստերների քանակը կարող եք փոփոխել
kmeans_labels = kmeans.fit_predict(X_scaled)

# PCA օգտագործում ենք՝ 2D տարածքում դիտելու համար
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Scatter plot կառուցելը՝ PCA-ի և KMeans կլաստերներով
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering Scatter Plot (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Ընտրում ենք հատկանիշները
X = df[['age', 'avg_glucose_level', 'bmi']]  # Կարող եք ավելացնել այլ հատկանիշներ

# Նորմալացնում ենք տվյալները
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans մոդել
kmeans = KMeans(n_clusters=3, random_state=42)  # Կլաստերների քանակը կարող եք փոփոխել
kmeans_labels = kmeans.fit_predict(X_scaled)

# 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot ըստ 3 հատկանիշների
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=kmeans_labels, cmap='viridis')

# Նկարագրություններ
ax.set_xlabel('Age')
ax.set_ylabel('Average Glucose Level')
ax.set_zlabel('BMI')
ax.set_title('K-Means Clustering 3D Scatter Plot')

plt.show()

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Ընտրում ենք կարևոր հատկանիշները
X = df[['age', 'avg_glucose_level', 'bmi']]

# Նորմալացնում ենք տվյալները
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
k_values = range(1, 11)

for k in k_values:  # Ներառենք 1-ից 10
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Գծապատկերն elbow method-ի համար
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-cluster Sum of Squares)')
plt.show()

# Բարձրագույն Silhouette Score ունեցող k
optimal_k = k_values[np.argmax(wcss)]
print(f"The optimal number of clusters (k) is: {optimal_k}")

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# Ընտրում ենք կարևոր հատկանիշները
X = df[['age', 'avg_glucose_level', 'bmi']]
# Նորմալացնում ենք տվյալները
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Silhouette Score հաշվարկում տարբեր k-ների համար
sil_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    sil_scores.append(score)

# Գծապատկեր Silhouette Score-ի համար
plt.figure(figsize=(8, 6))
plt.plot(k_values, sil_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette Score for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

# Բարձրագույն Silhouette Score ունեցող k
optimal_k = k_values[np.argmax(sil_scores)]
print(f"The optimal number of clusters (k) is: {optimal_k}")

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

X = df[['age', 'avg_glucose_level', 'bmi']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

agg_clust = AgglomerativeClustering(n_clusters=3, linkage='ward')  # 3 կլաստեր ընտրել
agg_labels = agg_clust.fit_predict(X_scaled)

# Dendrogram ստեղծելու համար
linked = linkage(X_scaled, method='ward')

# Dendrogram պլոտ
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agg_labels, cmap='viridis', marker='o')
plt.title('Hierarchical Clustering Scatter Plot')
plt.xlabel('Age')
plt.ylabel('Average Glucose Level')
plt.colorbar(label='Cluster')
plt.show()

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

# Silhouette Score-ի հաշվարկում
sil_scores = []
k_values = range(2, 11)

for k in k_values:
    hierarchical = AgglomerativeClustering(n_clusters=k)
    hierarchical.fit(X_scaled)
    score = silhouette_score(X_scaled, hierarchical.labels_)
    sil_scores.append(score)

plt.plot(k_values, sil_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette Score for Hierarchical Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

optimal_k = k_values[np.argmax(sil_scores)]
print(f"The optimal number of clusters (k) is: {optimal_k}")