import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset
df = pd.read_csv("data/telco.csv")

print("Dataset loaded!")
print("Dataset shape:", df.shape)

# Remove customer ID
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numbers
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df = df.dropna()

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# Find categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns

# Convert text columns to numbers automatically
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save complete pipeline
joblib.dump(pipeline, "models/churn_model.pkl")

print("Model saved successfully!")
print("Location: models/churn_model.pkl")