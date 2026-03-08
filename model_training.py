import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("patient_data.csv")

# 🔧 Clean stage labels FIRST
data['Stages'] = data['Stages'].str.strip()
data['Stages'] = data['Stages'].str.replace('.', '', regex=False)

data['Stages'] = data['Stages'].replace({
    'HYPERTENSIVE CRISISS': 'HYPERTENSIVE CRISIS',
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS'
})

# Check counts after cleaning
print(data['Stages'].value_counts())

# Features and target
X = data.drop('Stages', axis=1)
y = data['Stages']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:", X_train.shape[0])
print("Testing set:", X_test.shape[0])