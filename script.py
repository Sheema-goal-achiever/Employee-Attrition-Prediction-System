import pandas as pd
import kagglehub
import os
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- 1. DOWNLOAD DATA ---
path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
csv_path = os.path.join(path, [f for f in os.listdir(path) if f.endswith('.csv')][0])
df = pd.read_csv(csv_path)

# --- 2. CLEANING (MUST MATCH APP.PY) ---
# We ONLY drop the 4 columns that the app.py also ignores.
# We KEEP DailyRate, HourlyRate, and MonthlyRate because they are in the dataset.
cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df = df.drop(columns=cols_to_drop)

# --- 3. CATEGORICAL ENCODING ---
encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le 

# --- 4. PREPARE X and Y ---
# X must contain every column EXCEPT 'Attrition'
X = df.drop('Attrition', axis=1) 
y = df['Attrition']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE to balance the data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)



# --- 5. TRAIN MODEL ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# --- 6. SAVE ASSETS ---
# These files will now 'know' about DailyRate, HourlyRate, etc.
joblib.dump(model, 'attrition_model.pkl')
joblib.dump(encoders, 'encoders.pkl')

print("\n--- [SUCCESS] ---")
print(f"Model trained with {len(X.columns)} features.")
print("The error in app.py should now be gone.")