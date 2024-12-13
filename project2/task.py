import numpy as np
import pandas as pd

df = pd.read_csv('/home/rozale/Desktop/data_science_task/project2/economic_data_600k.csv')
df.head(20)
df.info()
df.isnull().sum()
df.isnull().sum()/len(df)*100
new_df = df.dropna()
new_df.head(20)
new_df.info()
df[df.duplicated()]
df['QuantitySold'].fillna(df['QuantitySold'].mean())
df.head(20)
df.describe()

# Counting Correlation
numeric_df = df.select_dtypes(include=[np.number])
print("Non-numeric columns:", df.select_dtypes(exclude=[np.number]).columns)
correlation_matrix = numeric_df.corr()
print(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(correlation_matrix, cmap = 'coolwarm', cbar = True)
plt.title('Corelation Matrix')
# plt.show()

subset = df[['QuantitySold', 'UnitPrice', 'TotalPrice']]
subset_corr = subset.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(subset_corr, annot=True, cmap='viridis', fmt='.2f', cbar=True)
plt.title("Correlation Matrix (Selected Features)")
# plt.show()

from sklearn.model_selection import train_test_split

# Կախյալ փոփոխական (Target) և անկախ փոփոխականներ (Features)
X = df.drop(columns=['TransactionID', 'TotalPrice'])  # Անկախ փոփոխականներ
y = df['TotalPrice']  # Կախյալ փոփոխական

# Տվյալների բաշխում ուսուցման և թեստի համար
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ստուգում, որ բաժանումը ճիշտ է կատարվել
print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)

import matplotlib.pyplot as plt

# `CustomerType` սյունի դասերի բաշխում ուսուցման ենթաբաժնում
plt.figure(figsize=(8, 4))
X_train['CustomerType'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Training Set Class Distribution (CustomerType)')
plt.xlabel('Customer Type')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
# plt.show()

# `CustomerType` սյունի դասերի բաշխում թեստային ենթաբաժնում
plt.figure(figsize=(8, 4))
X_test['CustomerType'].value_counts().plot(kind='bar', color=['blue', 'hotpink'])
plt.title('Test Set Class Distribution (CustomerType)')
plt.xlabel('Customer Type')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
# plt.show()

# `QuantitySold` սյունի բաշխում
bins = np.arange(0, df['QuantitySold'].max() + 10, 10)
df['QuantitySold'].hist(bins=bins, color='skyblue', edgecolor='black')
plt.title('Quantity Sold Distribution')
plt.xlabel('Quantity Sold')
plt.ylabel('Frequency')
plt.xticks(bins)
# plt.show()

# Տվյալների բաժանում ուսուցման և թեստի համար
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ստանդարտիզացում
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# # Ստանդարտիզացում ուսուցման տվյալների համար
# Exclude non-numeric columns before applying StandardScaler
numeric_columns = X.select_dtypes(include=[np.number]).columns

# Apply StandardScaler only on numeric columns
X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled = scaler.transform(X_test[numeric_columns])

# Replace the scaled values back into the DataFrame
X_train[numeric_columns] = X_train_scaled
X_test[numeric_columns] = X_test_scaled

# One-hot encoding categorical columns
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Ռեգրեսիոն մոդել
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ROC Curve (ռեգրեսիոն մոդելների համար այսպես չի կիրառվում, բայց եթե ուզում ես օգտագործել որպես այլ մեթոդ)
# from sklearn.metrics import roc_curve, auc
# fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test_scaled))
# roc_auc = auc(fpr, tpr)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict on the test data
y_pred = model.predict(X_test_scaled)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Line of perfect prediction
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
# plt.show()

# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score

# # Precision-Recall Curve
# precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
# average_precision = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])

# # Պատկերի վիզուալիզացիա
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, color='b', label='Precision-Recall curve (AP = %0.2f)' % average_precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc='lower left')
# plt.show()
import matplotlib.pyplot as plt

# Predict on the test data
y_pred = model.predict(X_test_scaled)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R²):", r2)

# Scatter plot of Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Line of perfect prediction
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
# plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Fill missing values in 'QuantitySold' column with the mean of the column
df['QuantitySold'] = df['QuantitySold'].fillna(df['QuantitySold'].mean())

# Description of numerical columns
print(df.describe())

# Correlation analysis
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
print(correlation_matrix)

# Visualize the correlation matrix
sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
# plt.show()

# Subset for specific features and visualize the correlation
subset = df[['QuantitySold', 'UnitPrice', 'TotalPrice']]
subset_corr = subset.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(subset_corr, annot=True, cmap='viridis', fmt='.2f', cbar=True)
plt.title("Correlation Matrix (Selected Features)")
# plt.show()

# Split data into features (X) and target (y)
X = df.drop(columns=['TransactionID', 'TotalPrice', 'CustomerType'])
y = df['CustomerType']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numeric features
scaler = StandardScaler()
numeric_columns = X.select_dtypes(include=[np.number]).columns
X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled = scaler.transform(X_test[numeric_columns])

# Replace the scaled values back into the DataFrame
X_train[numeric_columns] = X_train_scaled
X_test[numeric_columns] = X_test_scaled

# One-hot encode categorical columns
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Using SGDClassifier with partial_fit (for handling large datasets)
sgd_model = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=42)

# Train using partial_fit in batches
batch_size = 5000
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    sgd_model.partial_fit(X_batch, y_batch, classes=np.unique(y_train))

# Predict on the test data
y_pred_svm = sgd_model.predict(X_test)

# Evaluate SGDClassifier
accuracy_sgd = accuracy_score(y_test, y_pred_svm)
print("SGDClassifier Accuracy:", accuracy_sgd)

# Generate confusion matrix for SGDClassifier
cm_sgd = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_sgd, annot=True, fmt='d', cmap='Blues', xticklabels=['Business', 'Individual'], yticklabels=['Business', 'Individual'])
plt.title('Confusion Matrix for SGDClassifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# plt.show()

# K-Means
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Apply scaling to the features that will be used for clustering
X_for_clustering = df.drop(columns=['TransactionID', 'TotalPrice', 'CustomerType'])
numeric_columns = X_for_clustering.select_dtypes(include=[np.number]).columns
X_for_clustering[numeric_columns] = StandardScaler().fit_transform(X_for_clustering[numeric_columns])

# # Apply KMeans Clustering
# 1. Ոչ թվային սյուների հայտնաբերում
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
print("Ոչ թվային սյուներ:", non_numeric_columns)

# Օրացուցային կամ տեքստային սյուների մշակում
for col in non_numeric_columns:
    if 'date' in col.lower() or 'DateColumn' in col:  # Նույնացնում ենք օրացուցային սյուները
        df[col] = pd.to_datetime(df[col])  # Վերածում datetime-ի
        df[f'{col}_Year'] = df[col].dt.year
        df[f'{col}_Month'] = df[col].dt.month
        df[f'{col}_Day'] = df[col].dt.day
        df = df.drop(columns=[col])  # Հեռացնում ենք օրացուցային սյունը
    else:
        # Տեքստային սյուների փոխակերպում թվային ձևաչափի (LabelEncoder կամ get_dummies)
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# 2. Դատարկ արժեքների մշակում
print("Դատարկ արժեքներ՝ նախքան մշակում:", df.isnull().sum())
df = df.fillna(df.mean())
print("Դատարկ արժեքներ՝ մշակումից հետո:", df.isnull().sum())

# 3. Կլաստերացման տվյալների պատրաստում
columns_to_drop = ['TransactionID', 'TotalPrice', 'CustomerType']
columns_to_drop = [col for col in columns_to_drop if col in df.columns]  # Պայմանական ստուգում
X_for_clustering = df.drop(columns=columns_to_drop)

# Ստանդարտիզացում (միայն թվային սյուների համար)
numeric_columns = X_for_clustering.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_for_clustering[numeric_columns] = scaler.fit_transform(X_for_clustering[numeric_columns])

# 4. KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Որոշում ենք 3 կլաստեր
df['Cluster'] = kmeans.fit_predict(X_for_clustering)

# 5. Վիզուալիզացիա
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='QuantitySold', y='UnitPrice', hue='Cluster', palette='viridis', s=100)
plt.title('K-Means Clustering Results (Quantity Sold vs Unit Price)')
plt.xlabel('Quantity Sold')
plt.ylabel('Unit Price')
plt.legend(title='Cluster')
# plt.show()

# Կլաստերների կենտրոնների ցուցադրում
cluster_centers = kmeans.cluster_centers_
plt.figure(figsize=(8, 6))
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
plt.scatter(X_for_clustering['QuantitySold'], X_for_clustering['UnitPrice'], s=50, c=df['Cluster'], cmap='viridis', alpha=0.7)
plt.title('K-Means Clustering with Cluster Centers')
plt.xlabel('Quantity Sold')
plt.ylabel('Unit Price')
plt.legend()
# plt.show()

# Կլաստերների ամփոփ տվյալներ
cluster_summary = df.groupby('Cluster').agg({
    'QuantitySold': ['mean', 'std'],
    'UnitPrice': ['mean', 'std'],
    'TotalPrice': ['mean', 'std']
}).reset_index()
print("Կլաստերների ամփոփում:")
print(cluster_summary)


# # կոռելյացիայի ցուցադրում
# # Alternative 1: Using Logistic Regression
# # Confusion matrix for Logistic Regression

# # Alternative 2: Using RandomForestClassifier
# # Confusion matrix for Random Forest
