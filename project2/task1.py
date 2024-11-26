# Անհրաժեշտ է կիրառել մի քանի supervised ML մոդելներ.

# ✅️ գնահատել ճշտությունը, 
# ✅️ համեմատել, 
# ✅️ կառուցել մոդելի scatter գրաֆիկը, 
# ✅️ ինչպես նաև accuracy ու error rate գրաֆիկները։ 

# Այս ամենից հետո ջնջել supervised սարքող մասը (y կամ target...), 
# կրկնել նույնը unsupervised ML մոդելներով.
# ✅️ գնահատել, 
# ✅️ համեմատել, 
# ✅️ գրաֆիկներ կառուցել։ 

# Database ֊ ը պետք ա հնարավորինս մեծ լինի, միլիոնը պարտադիր չի։

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('/home/rozale/Desktop/DS/diabetic_data.csv')

# Handle missing values
# Fill missing values in numeric columns with the mean
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# For categorical columns (e.g., gender, race), fill missing values with the mode or a placeholder
categorical_columns = data.select_dtypes(include=[object]).columns
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Label encode categorical variables
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['race'] = label_encoder.fit_transform(data['race'])
data['readmitted'] = label_encoder.fit_transform(data['readmitted'])

# Define features and target variable for supervised models
X = data.drop(columns=['readmitted', 'encounter_id', 'patient_nbr'])  # Remove non-feature columns
y = data['readmitted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate supervised models
model_results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_results[model_name] = accuracy
    print(f'{model_name} Accuracy: {accuracy}')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.title(f'{model_name} Confusion Matrix')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(2), ['Not Readmitted', 'Readmitted'])
    plt.yticks(np.arange(2), ['Not Readmitted', 'Readmitted'])
    plt.show()

    # Scatter plot for predictions vs actual values (for model comparison)
    plt.scatter(y_test, y_pred)
    plt.title(f'{model_name} Predictions vs Actuals')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

    # Accuracy and error rate graphs
    errors = 1 - accuracy
    plt.bar([model_name], [accuracy], label="Accuracy", color='blue')
    plt.bar([model_name], [errors], label="Error Rate", color='red')
    plt.ylabel('Rate')
    plt.legend()
    plt.show()

# Unsupervised Models (without target variable)
X_unsupervised = data.drop(columns=['encounter_id', 'patient_nbr', 'readmitted'])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_unsupervised)
data['cluster'] = kmeans.labels_

# Silhouette Score
sil_score = silhouette_score(X_unsupervised, kmeans.labels_)
print(f'K-Means Silhouette Score: {sil_score}')

# PCA for dimensionality reduction (for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_unsupervised)

# Scatter plot for PCA results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('PCA of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
