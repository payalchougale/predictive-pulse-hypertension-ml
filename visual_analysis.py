import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
data = pd.read_csv("patient_data.csv")

# 🔧 Clean stage labels (FIX FOR GRAPH + MODEL)
data['Stages'] = data['Stages'].str.strip()

data['Stages'] = data['Stages'].replace({
    'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)',
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS',
    'HYPERTENSIVE CRISISS': 'HYPERTENSIVE CRISIS'
})

sns.set(style="whitegrid")

# Create folder for graphs
os.makedirs("graphs", exist_ok=True)

# 1️⃣ Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='C', data=data)
plt.title("Gender Distribution")
plt.savefig("graphs/gender_countplot.png")
plt.show()

data['C'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Gender Distribution Pie Chart")
plt.ylabel("")
plt.savefig("graphs/gender_piechart.png")
plt.show()


# 2️⃣ Hypertension Stages Distribution
plt.figure(figsize=(8,5))

ax = sns.countplot(x='Stages', hue='Stages', data=data, palette="Set2", legend=False)

plt.title("Hypertension Stages Distribution", fontsize=14)
plt.xlabel("Hypertension Stage")
plt.ylabel("Number of Patients")
plt.xticks(rotation=30)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig("graphs/hypertension_stages.png")
plt.show()


# Convert BP ranges to numeric
data['Systolic'] = data['Systolic'].astype(str).str.extract('(\d+)').astype(float)
data['Diastolic'] = data['Diastolic'].astype(str).str.extract('(\d+)').astype(float)


# 3️⃣ Correlation Heatmap
plt.figure(figsize=(8,6))
corr_matrix = data[['Systolic','Diastolic']].corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")

plt.title("Correlation between Systolic and Diastolic Pressure")
plt.savefig("graphs/correlation_heatmap.png")
plt.show()


# 4️⃣ TakeMedication vs Severity
plt.figure(figsize=(6,4))
sns.countplot(x='Severity', hue='TakeMedication', data=data)
plt.title("Medication vs Severity")
plt.savefig("graphs/medication_vs_severity.png")
plt.show()


# 5️⃣ Age vs Hypertension Stage
plt.figure(figsize=(6,4))
sns.countplot(x='Age', hue='Stages', data=data)
plt.title("Age Group vs Hypertension Stage")
plt.savefig("graphs/age_vs_stage.png")
plt.show()


# 6️⃣ Pairplot
pair = sns.pairplot(data[['Systolic','Diastolic','Stages']], hue='Stages')
pair.savefig("graphs/pairplot.png")
plt.show()