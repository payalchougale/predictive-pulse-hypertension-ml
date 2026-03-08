import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
data = pd.read_csv("patient_data.csv")

# Create Label Encoder
le = LabelEncoder()

# Columns to encode
columns = [
    'C','Age','History','Patient','TakeMedication','Severity',
    'BreathShortness','VisualChanges','NoseBleeding',
    'Whendiagnoused','Systolic','Diastolic','ControlledDiet','Stages'
]

# Apply Label Encoding
for col in columns:
    data[col] = le.fit_transform(data[col])

print("Encoded Dataset:")
print(data.head())


scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

data = pd.DataFrame(scaled_data, columns=data.columns)

print("\nScaled Dataset:")
print(data.head())

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

data = pd.DataFrame(scaled_data, columns=data.columns)

print("\nScaled Dataset:")
print(data.head())