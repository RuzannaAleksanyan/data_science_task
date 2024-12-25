import pandas as pd
df = pd.read_csv('/home/rozale/Desktop/data_science_task/project2/economic_data_600k.csv')
df.info()
print(df.head(20))
print(df.isnull().sum())
new_df = df.dropna()
print(new_df.head(20))
new_df.info()
df['QuantitySold'].fillna(df['QuantitySold'].mean())
print(df.head(20))
df.describe()

# Կորելացիայի մատրից թվային սյունակների միջև
import numpy as np
numeric_df = df.select_dtypes(include=[np.number])
print("Non-numeric columns:", df.select_dtypes(exclude=[np.number]).columns)
correlation_matrix = numeric_df.corr()
print("Correlation matrix: ")
print(correlation_matrix)

# Կորելացիոն մատրիցի ցուցադրում գրաֆիկորեն
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(correlation_matrix, cmap = 'coolwarm', cbar = True)
plt.title('Corelation Matrix')
plt.show()
# Կորելացիոն մատրիցի ցուցադրում ընտրված սյունակների միջև
subset = df[['QuantitySold', 'UnitPrice', 'TotalPrice']]
subset_corr = subset.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(subset_corr, annot=True, cmap='viridis', fmt='.2f', cbar=True)
plt.title("Correlation Matrix (Selected Features)")
plt.show()

from sklearn.model_selection import train_test_split
X = df.drop(columns=['TransactionID', 'TotalPrice'])  # Feature-ներից հեռացնում ենք ID և թիրախ սյունակները։
y = df['TotalPrice']  # Թիրախ փոփոխականը սահմանում ենք որպես TotalPrice։
# Տվյալները բաժանում ենք ուսուցման (80%) և թեստային (20%) հավաքածուների։
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)

# CustomerType սյունակի հաճախականություն train collection-ում
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
X_train['CustomerType'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Training Set Class Distribution (CustomerType)')
plt.xlabel('Customer Type')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()

# QuantitySold սյունի հիստոգրամ
bins = np.arange(0, df['QuantitySold'].max() + 10, 10)
df['QuantitySold'].hist(bins=bins, color='skyblue', edgecolor='black')
plt.title('Quantity Sold Distribution')
plt.xlabel('Quantity Sold')
plt.ylabel('Frequency')
plt.xticks(bins)
plt.show()

# Ստանդարտիզացում
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_columns = X.select_dtypes(include=[np.number]).columns

# Թվային սյուների միջին և ստանդարտ շեղում
X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled = scaler.transform(X_test[numeric_columns])
# Ստանդարտիզացված արժեքների վերագրում
X_train[numeric_columns] = X_train_scaled
X_test[numeric_columns] = X_test_scaled
# X_train֊ի և X_test֊ի կոդավորում
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Ռեգրեսիոն մոդել
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# Trined LinearRegression
model.fit(X_train_scaled, y_train)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = model.predict(X_test_scaled)
# միջին քառակուսային սխալ
mse = mean_squared_error(y_test, y_pred)
# միջին բացարձակ սխալ
mae = mean_absolute_error(y_test, y_pred)
# R^2 գնահատական (0, 1)(համապատասխանություն)
r2 = r2_score(y_test, y_pred)

# Իրական և կանխատեսվող արժեքների գրաֆիկ
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# SGDRegressor մոդել
sgd_model = SGDRegressor(max_iter=1000, random_state=42)

batch_size = 5000
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    sgd_model.partial_fit(X_batch, y_batch)

# y արժեքի կանխատեսում
y_pred_sgd = sgd_model.predict(X_test)

# Միջին քառակուսային սխալի հաշվում
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
# Միջին բացարձակ սխալի հաշվում
mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
# R^2 գնահատական
r2_sgd = r2_score(y_test, y_pred_sgd)

print("SGDRegressor MSE:", mse_sgd)
print("SGDRegressor MAE:", mae_sgd)
print("SGDRegressor R^2:", r2_sgd)

# SGDRegressor մեդելի վիզուալիզացիա
plt.scatter(y_test, y_pred_sgd, alpha=0.5, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values (SGDRegressor)')
plt.show()

# Clustering - K-Means 
from sklearn.cluster import KMeans

X_for_clustering = df.drop(columns=['TransactionID', 'TotalPrice', 'CustomerType'])
numeric_columns = X_for_clustering.select_dtypes(include=[np.number]).columns
X_for_clustering[numeric_columns] = StandardScaler().fit_transform(X_for_clustering[numeric_columns])

non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns:", non_numeric_columns)

# Ոչ թվային սյուների մշակում
for col in non_numeric_columns:
    if 'date' in col.lower() or 'DateColumn' in col:  
        df[col] = pd.to_datetime(df[col]) 
        df[f'{col}_Year'] = df[col].dt.year
        df[f'{col}_Month'] = df[col].dt.month
        df[f'{col}_Day'] = df[col].dt.day
        df = df.drop(columns=[col]) 
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Դատարկ արժեքների մշակում
print("Empty values before processing:", df.isnull().sum())
df = df.fillna(df.mean())
print("Empty values after processing:", df.isnull().sum())

# Կլաստերացման տվյալների պատրաստում
columns_to_drop = ['TransactionID', 'TotalPrice', 'CustomerType']
columns_to_drop = [col for col in columns_to_drop if col in df.columns] 
X_for_clustering = df.drop(columns=columns_to_drop)

# Ստանդարտիզացում (միայն թվային սյուների համար)
numeric_columns = X_for_clustering.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_for_clustering[numeric_columns] = scaler.fit_transform(X_for_clustering[numeric_columns])

# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)  # Որոշում ենք 3 կլաստեր
df['Cluster'] = kmeans.fit_predict(X_for_clustering)

# Վիզուալիզացիա
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='QuantitySold', y='UnitPrice', hue='Cluster', palette='viridis', s=100)
plt.title('K-Means Clustering Results (Quantity Sold vs Unit Price)')
plt.xlabel('Quantity Sold')
plt.ylabel('Unit Price')
plt.legend(title='Cluster', loc='upper right') 
plt.show()

# # Կլաստերների կենտրոնների ցուցադրում
# cluster_centers = kmeans.cluster_centers_
# plt.figure(figsize=(8, 6))
# plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
# plt.scatter(X_for_clustering['QuantitySold'], X_for_clustering['UnitPrice'], s=50, c=df['Cluster'], cmap='viridis', alpha=0.7)
# plt.title('K-Means Clustering with Cluster Centers')
# plt.xlabel('Quantity Sold')
# plt.ylabel('Unit Price')
# plt.legend(title='Cluster', loc='upper right')
# plt.show()

# Կլաստերների ամփոփ տվյալներ
cluster_summary = df.groupby('Cluster').agg({
    'QuantitySold': ['mean', 'std'],
    'UnitPrice': ['mean', 'std'],
    'TotalPrice': ['mean', 'std']
}).reset_index()
print("Cluster summary:")
print(cluster_summary)
