import pandas as pd

data = pd.read_csv("patient_data.csv")

print(data.head())
print(data.shape)
print(data.columns)