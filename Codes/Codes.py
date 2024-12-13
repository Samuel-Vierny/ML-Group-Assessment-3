import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import numpy as np

# Load dataset using the provided file path
heart_disease_data = pd.read_csv('H:\\Samuel\'s folder\\2024 - OPIT\\Assessments\\ML\\Assessment 3 ML - Group Project\\Data\\heart_disease_cleaned.csv')

# Display the first few rows to confirm successful loading
print(heart_disease_data.head())

# Separate features (X) and target (y)
X = heart_disease_data.drop(columns=['id', 'num'])  # Dropping ID and target columns
y = heart_disease_data['num']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Preprocessing pipelines
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

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Train and evaluate supervised models
model_results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    # Train the model
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

# Perform K-means clustering
pipeline_kmeans = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=2, random_state=42))
])

pipeline_kmeans.fit(X_train)
clusters = pipeline_kmeans.named_steps['kmeans'].labels_
silhouette_avg = silhouette_score(pipeline_kmeans.named_steps['preprocessor'].transform(X_train), clusters)

# Output results
print("Supervised Model Results:")
for name, results in model_results.items():
    print(f"\n{name}:\n")
    print(f"Classification Report:\n{results['classification_report']}")
    print(f"ROC-AUC Score: {results['roc_auc_score']}")

print("\nUnsupervised Model Results:")
print(f"K-means Silhouette Score: {silhouette_avg}")