import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans

# Տվյալների ներբեռնում
file_path = "linear_regression_dataset.csv"  # Գեներացված ֆայլի տեղադրությունը
df = pd.read_csv(file_path)

# Նախնական տվյալների նկարագրություն
print("Dataset Info:")
df.info()
print("\nDataset Head:")
print(df.head(20))
print("\nMissing values per column:")
print(df.isnull().sum())

# Կորելացիայի մատրից
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Matrix')
plt.show()

# X, y առանձնացում
X = df.drop(columns=['target'])
y = df['target']

# Տվյալների բաժանում ուսուցման և թեստավորման հավաքածուների
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ստանդարտիզացում
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression մոդելի ուսուցում
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Գնահատում
y_pred_lr = lr_model.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression MSE: {mse_lr}")
print(f"Linear Regression MAE: {mae_lr}")
print(f"Linear Regression R^2: {r2_lr}")

# Linear Regression արդյունքների գրաֆիկ
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()

# SGD Regressor մոդելի ուսուցում
sgd_model = SGDRegressor(max_iter=1000, random_state=42)
sgd_model.fit(X_train_scaled, y_train)

# Գնահատում
y_pred_sgd = sgd_model.predict(X_test_scaled)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

print(f"SGD Regressor MSE: {mse_sgd}")
print(f"SGD Regressor MAE: {mae_sgd}")
print(f"SGD Regressor R^2: {r2_sgd}")

# SGD Regressor արդյունքների գրաֆիկ
plt.scatter(y_test, y_pred_sgd, alpha=0.5, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('SGD Regressor: Actual vs Predicted')
plt.show()

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Կլաստերացման արդյունքների վիզուալիզացիա
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='feature_1', y='feature_2', hue='Cluster', palette='viridis', s=100)
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(title='Cluster', loc='upper right')
plt.show()

# Կլաստերների ամփոփ տվյալներ
cluster_summary = df.groupby('Cluster').agg({
    'feature_1': ['mean', 'std'],
    'feature_2': ['mean', 'std'],
    'feature_3': ['mean', 'std'],
    'target': ['mean', 'std']
}).reset_index()
print("Cluster Summary:")
print(cluster_summary)
