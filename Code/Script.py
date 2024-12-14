import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.utils import resample  
from sklearn.cluster import KMeans
import shap 

# Dynamically determine the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the dataset file path relative to the script's directory
file_name = 'heart_disease_cleaned.csv'
file_path = os.path.join(script_dir, file_name)

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"Dataset '{file_name}' not found in the script directory: {script_dir}")

# Load the dataset
print(f"Loading dataset from: {file_path}")

heart_disease_data = pd.read_csv(file_path)

#Data Preparation
# Separate features (X) and target (y)
X = heart_disease_data.drop(columns=['id', 'num'])
y = heart_disease_data['num']

# Handle Class Imbalance with Re-sampling
X, y = resample(X, y, replace=True, stratify=y, random_state=42)  # Over-sample minority classes

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Preprocessing Pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Supervised Learning Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

model_results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Collect metrics
    metrics = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro') if y_pred_proba is not None else None
    
    model_results[name] = {
        "classification_report": metrics,
        "roc_auc_score": auc
    }
    
    # Step 4: SHAP for Model Interpretability
    if name == "Random Forest":  # Example for Random Forest
        # Ensure the transformed dataset is used
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

        # SHAP values calculation for Random Forest
        explainer = shap.TreeExplainer(pipeline.named_steps['model'])
        shap_values = explainer.shap_values(X_test_transformed)
        
        # Ensure the correct dataset is passed to summary_plot
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        shap.summary_plot(shap_values, pd.DataFrame(X_test_transformed, columns=feature_names), plot_type="bar")


# Step 5: Unsupervised Learning
# Dimensionality Reduction (Optional)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

# Preprocess the data before PCA
X_train_transformed = preprocessor.transform(X_train)

# Apply PCA on the preprocessed data
X_train_pca = pca.fit_transform(X_train_transformed)


# K-Means Clustering
pipeline_kmeans = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])
pipeline_kmeans.fit(X_train)
clusters = pipeline_kmeans.named_steps['kmeans'].labels_
silhouette_avg = silhouette_score(pipeline_kmeans.named_steps['preprocessor'].transform(X_train), clusters)

# Step 6: Output Results
print("Supervised Model Results:")
for name, results in model_results.items():
    print(f"\n{name}:\n")
    print(f"Classification Report:\n{results['classification_report']}")
    print(f"ROC-AUC Score: {results['roc_auc_score']}")

print("\nUnsupervised Model Results:")
print(f"K-means Silhouette Score: {silhouette_avg:.4f}")
