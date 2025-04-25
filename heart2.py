import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("Dataset/heart_disease_data.csv")

# Categorize 'cp' (Chest Pain) into two groups: 0 (Less Severe), 1 (More Severe)
df['cp'] = df['cp'].apply(lambda x: 0 if x in [0, 1] else 1)

# Separate features and target
X = df.drop(columns=["target"])  # Assuming 'target' is the outcome column
y = df["target"]

# Handle missing values (replace NaN with column median)
X.fillna(X.median(), inplace=True)
print("Features used in training:", list(X.columns))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the correct model (removed the tuple issue)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save the model and scaler
joblib.dump(model, "heart_disease_model3.sav")
joblib.dump(scaler, "scaler3.sav")
