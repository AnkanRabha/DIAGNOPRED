import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
import joblib
# Load dataset
file_path = "Dataset/Indian Liver Patient Dataset (ILPD).csv"
df = pd.read_csv(file_path, header=None)

# Rename columns based on dataset documentation
df.columns = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphatase",
    "Alanine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Proteins", 
    "Albumin", "Albumin_to_Globulin_Ratio", "Outcome"
]

# Convert categorical "Gender" column (Male = 1, Female = 0)
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Handle missing values (Fill NaN with median)
df.fillna(df.median(), inplace=True)

# Winsorize outliers (capping extreme values)
winsor_columns = ["Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphatase", 
                  "Alanine_Aminotransferase", "Aspartate_Aminotransferase"]
for col in winsor_columns:
    df[col] = winsorize(df[col], limits=[0.05, 0.05])

# Recode the 'Outcome' column to ensure binary class labels (0, 1)
df["Outcome"] = df["Outcome"].map({1: 1, 2: 0})

# Define features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for balancing the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Initialize models

XGB= XGBClassifier()



XGB.fit(X_train, y_train)
y_pred = XGB.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"{XGB} accuracy: {accuracy:.4f}")

# joblib.dump(XGB, "liver_model.pkl")
joblib.dump(scaler, "liver_scaler.pkl")