# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
# Load dataset
df = pd.read_csv("Dataset/stroke.csv")

# Drop 'id' and unnecessary categorical columns
df.drop(columns=['id', 'ever_married', 'work_type', 'Residence_type'], inplace=True)

# Fill missing values in 'bmi' with the median
df.loc[:, 'bmi'] = df['bmi'].fillna(df['bmi'].median())


# Encode categorical variables
label_encoders = {}
for col in ['gender', 'smoking_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for future decoding if needed

# Split data into features and target
X = df.drop(columns=['stroke'])
y = df['stroke']

# Apply SMOTE for oversampling the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert resampled data back to a DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['stroke'] = y_resampled  # Add target column back

# Check new class distribution
# print("Class distribution after SMOTE:")
# print(y_resampled.value_counts(normalize=True) * 100)


# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


RF= RandomForestClassifier(n_estimators=100, random_state=42)



RF.fit(X_train, y_train)  # Train model
y_pred = RF.predict(X_test)  # Predict on test data
    
    # Compute performance metrics
metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    
 

# Convert results to DataFrame for easy comparison
performance_df = pd.DataFrame([metrics])
print(performance_df)

joblib.dump(RF, "stroke_model.pkl")
