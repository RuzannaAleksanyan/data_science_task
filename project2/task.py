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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

# Load the dataset
file_path = 'diabetic_data.csv'  # Update with the path to your file
data = pd.read_csv(file_path)

# Preprocessing
data.replace('?', pd.NA, inplace=True)  # Replace '?' with NaN
columns_to_drop = ['max_glu_serum', 'A1Cresult', 'weight', 'payer_code', 'medical_specialty']
data_cleaned = data.drop(columns=columns_to_drop)  # Drop columns with excessive missing values

# Encode categorical variables
categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col, encoder in label_encoders.items():
    data_cleaned[col] = encoder.fit_transform(data_cleaned[col].astype(str))

# Select target and features
target_column = 'readmitted'  # Target variable for classification
X = data_cleaned.drop(columns=[target_column])
y = data_cleaned[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models with optimized parameters
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    "Support Vector Machine": SVC(kernel="linear", C=0.1, random_state=42)  # Simplified SVM
}

# Train models and evaluate performance
model_performance = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    start_time = time.time()  # Start timing
    
    model.fit(X_train_scaled, y_train)  # Train the model
    
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"{model_name} training completed in {elapsed_time:.2f} seconds.")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate accuracy and error rate
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    
    # Print results
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Error Rate: {error_rate:.4f}")
    
    # Store results
    model_performance[model_name] = {"Accuracy": accuracy, "Error Rate": error_rate}

# Convert results to a DataFrame for visualization
performance_df = pd.DataFrame(model_performance).T

# Plot accuracy and error rate
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy plot
performance_df["Accuracy"].plot(kind="bar", ax=axes[0], color="skyblue")
axes[0].set_title("Model Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
axes[0].set_xticklabels(performance_df.index, rotation=45)

# Error rate plot
performance_df["Error Rate"].plot(kind="bar", ax=axes[1], color="salmon")
axes[1].set_title("Model Error Rate")
axes[1].set_ylabel("Error Rate")
axes[1].set_ylim(0, 1)
axes[1].set_xticklabels(performance_df.index, rotation=45)

plt.tight_layout()
plt.show()

# Display performance metrics
print(performance_df)



print("....................")
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Drop the target variable for unsupervised learning
X_unsupervised = data_cleaned.drop(columns=[target_column])

# Standardize the features
X_scaled = scaler.fit_transform(X_unsupervised)

# Initialize unsupervised models
unsupervised_models = {
    "K-Means": KMeans(n_clusters=3, random_state=42),
    "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
    "Agglomerative Clustering": AgglomerativeClustering(n_clusters=3)
}

# Train and evaluate unsupervised models
unsupervised_performance = {}

for model_name, model in unsupervised_models.items():
    print(f"Training {model_name}...")
    
    # Train model
    start_time = time.time()
    labels = model.fit_predict(X_scaled)
    elapsed_time = time.time() - start_time
    print(f"{model_name} training completed in {elapsed_time:.2f} seconds.")
    
    # Evaluate model performance
    silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else float("nan")
    davies_bouldin = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else float("nan")
    
    print(f"{model_name} - Silhouette Score: {silhouette:.4f}, Davies-Bouldin Score: {davies_bouldin:.4f}")
    
    # Store results
    unsupervised_performance[model_name] = {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Score": davies_bouldin
    }

# Convert results to a DataFrame for visualization
unsupervised_performance_df = pd.DataFrame(unsupervised_performance).T

# Plot performance metrics
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Silhouette Score plot
unsupervised_performance_df["Silhouette Score"].plot(kind="bar", ax=axes[0], color="lightgreen")
axes[0].set_title("Silhouette Score")
axes[0].set_ylabel("Score")
axes[0].set_xticklabels(unsupervised_performance_df.index, rotation=45)

# Davies-Bouldin Score plot
unsupervised_performance_df["Davies-Bouldin Score"].plot(kind="bar", ax=axes[1], color="orange")
axes[1].set_title("Davies-Bouldin Score")
axes[1].set_ylabel("Score")
axes[1].set_xticklabels(unsupervised_performance_df.index, rotation=45)

plt.tight_layout()
plt.show()

# Scatter plot visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, len(unsupervised_models), figsize=(15, 5))
for i, (model_name, model) in enumerate(unsupervised_models.items()):
    labels = model.fit_predict(X_scaled)
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=10)
    axes[i].set_title(f"{model_name} Clustering")
    axes[i].set_xlabel("PCA Component 1")
    axes[i].set_ylabel("PCA Component 2")

plt.tight_layout()
plt.show()

# Display performance metrics
print(unsupervised_performance_df)
