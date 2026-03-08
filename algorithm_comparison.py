import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib


# Graph style
sns.set(style="whitegrid")

# Load dataset
data = pd.read_csv("patient_data.csv")

# 🔧 Clean BP values
data['Systolic'] = data['Systolic'].astype(str).str.extract(r'(\d+)').astype(float)
data['Diastolic'] = data['Diastolic'].astype(str).str.extract(r'(\d+)').astype(float)

# Fix stage names
data['Stages'] = data['Stages'].replace({
    'HYPERTENSIVE CRISISS': 'HYPERTENSIVE CRISIS',
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS'
})

# Encode categorical columns
categorical_cols = data.select_dtypes(include=['object','string']).columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Split features and target
X = data.drop('Stages', axis=1)
y = data['Stages']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Feature scaling
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Ridge Classifier": RidgeClassifier(),
    "Gaussian Naive Bayes": GaussianNB()
}

# Store results
results = {}

print("\n🔹 Model Performance Results 🔹\n")

# Train and evaluate models
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy * 100

    print("\n==============================")
    print(f"{name}")
    print("==============================")

    print(f"Accuracy: {accuracy*100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# Convert results to dataframe
results_df = pd.DataFrame(list(results.items()), columns=['Algorithm', 'Accuracy'])

# Plot comparison graph
plt.figure(figsize=(10,6))

sns.barplot(x='Algorithm', y='Accuracy', data=results_df)

plt.title("Machine Learning Algorithm Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("Algorithms")

plt.xticks(rotation=45)

# Add accuracy labels
for index, row in results_df.iterrows():
    plt.text(index, row.Accuracy + 0.5, f"{row.Accuracy:.1f}%", ha='center')

plt.tight_layout()

plt.savefig("algorithm_comparison_graph.png")

plt.show()

# Best algorithm
best_model = results_df.loc[results_df['Accuracy'].idxmax()]

print("\n⭐ Best Performing Model:")
print(best_model)

import joblib

# Train final Logistic Regression model
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)

# Save model
joblib.dump(logreg, "logreg_model.pkl")

print("Model saved as logreg_model.pkl")