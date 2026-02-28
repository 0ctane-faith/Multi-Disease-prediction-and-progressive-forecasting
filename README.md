import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Download dataset
path = kagglehub.dataset_download("daspias/multiple-disease-dataset")
print("Dataset path:", path)

# Find CSV
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found.")

dataset_csv_path = os.path.join(path, csv_files[0])
print("Using file:", dataset_csv_path)

# Load data
df = pd.read_csv(dataset_csv_path)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values before filling:")
print(df.isnull().sum())

# Fill missing numeric values (though no numeric columns are present in this dataset for now)
df = df.fillna(df.mean(numeric_only=True))

print("\nMissing values after filling:")
print(df.isnull().sum())

# --------- TARGET DETECTION ---------

possible_targets = ['target','disease','outcome','label']

target_column_name = None
for col in possible_targets:
    if col in df.columns:
        target_column_name = col
        break

if target_column_name is None:
    target_column_name = df.columns[-1]
    print(f"\n⚠ Target not found. Using last column: {target_column_name}")

print("Target column:", target_column_name)

# --------- FEATURE SELECTION ---------

X = df.drop(columns=[target_column_name])
y = df[target_column_name]

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include='object').columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create a column transformer for preprocessing
# One-hot encode categorical features, pass numerical features directly
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# --------- TRAIN TEST SPLIT ---------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------- MODEL TRAINING ---------

# Create a pipeline that first preprocesses the data then trains the classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

model.fit(X_train, y_train)

# --------- PREDICTION ---------

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------- CONFUSION MATRIX ---------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------- SAMPLE NEW PREDICTION ---------

# To predict on new data, it also needs to go through the same preprocessing pipeline
# We can't directly use X.mean() if X contains categorical data.
# For demonstration, let's create a sample row with the most frequent values for categorical columns
# and the mean for numerical (if any).

# Create a dummy new data point with the most frequent values for categorical features
# and average for numerical features (if any).
# This assumes X is the original DataFrame with mixed types.

# Create a dictionary to hold sample data
sample_data_dict = {}
for col in categorical_features:
    sample_data_dict[col] = [X[col].mode()[0]] # Use mode for categorical
for col in numerical_features:
    sample_data_dict[col] = [X[col].mean()] # Use mean for numerical

# Convert to DataFrame, ensuring column order matches X_train if not using pipeline directly
new_data_df = pd.DataFrame(sample_data_dict)

if not new_data_df.empty:
    prediction = model.predict(new_data_df)
    print("\nPrediction for sample patient:", prediction)
else:
    print("No usable features found for sample prediction.")
