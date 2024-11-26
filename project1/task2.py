import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('/home/rozale/Desktop/python/diabetic_data.csv')

df['age_numeric'] = df['age'].apply(lambda x: int(x.split('-')[1].replace(')', '').strip()) if '-' in x else int(x))

# 1 Summary of the DataFrame 'df'
df.info()

# 2 Count missing values in each column
missing_values = df.isnull().sum()
print(missing_values)
total_missing = df.isnull().sum().sum()
print(f"\nTotal missing values in the dataset: {total_missing}")

# 3 min max mod
weight_column = df['weight']

min_weight = weight_column.min()
max_weight = weight_column.max()
mode_weight = weight_column.mode().values

print(f"Minimum: {min_weight}")
print(f"Maximum: {max_weight}")
print(f"Mode: {mode_weight}")

# 4 Replace missing values with the maximum value of numeric columns
for column in df.columns:
    if df[column].isnull().any() and pd.api.types.is_numeric_dtype(df[column]):
        max_value = df[column].max()
        df[column].fillna(max_value, inplace=True)

print("\nMissing values after replacement:")
print(df.isnull().sum())

output_file_path = 'updated_dataset.csv'  # Specify the output file name
df.to_csv(output_file_path, index=False)

print(f"\nUpdated dataset saved to {output_file_path}")


# 5 Gender Distribution  
gender_counts = df['gender'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'salmon'])
plt.title('Gender Distribution')
plt.axis('equal') 
plt.show()


# 6 խմբավորել տվյալները ըստ «սեռ»֊ի և հաշվարկեք միջին քաշը:
df['weight'] = df['weight'].replace('?', np.nan)

def convert_weight_range_to_midpoint(weight):
    if pd.isnull(weight): 
        return np.nan
    if '-' in weight:
        numbers = weight.replace('[', '').replace(')', '').split('-')
        return (float(numbers[0]) + float(numbers[1])) / 2
    return np.nan 

df['weight'] = df['weight'].apply(convert_weight_range_to_midpoint)

df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

average_weight_by_race = df.groupby('race')['weight'].mean().reset_index()

average_weight_by_race.columns = ['Race', 'Average_Weight']

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(average_weight_by_race['Race'], average_weight_by_race['Average_Weight'], color='skyblue')
plt.title('Average Weight by Race')
plt.xlabel('Race')
plt.ylabel('Average Weight')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout() 
plt.show()

# 7 Scatter Plot of Age vs Time in Hospital
plt.figure(figsize=(10, 6))
plt.scatter(df['age_numeric'], df['time_in_hospital'], alpha=0.5, color='b')
plt.title('Scatter Plot of Age vs Time in Hospital')
plt.xlabel('Age')
plt.ylabel('Time in Hospital')
plt.grid(True)
plt.show()

# 8 Histograms for age distribution
plt.figure(figsize=(10, 6))
plt.hist(df['age_numeric'], bins=10, color='skyblue', edgecolor='black')

plt.title('Histogram of Age', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.grid(True)
plt.show()

# 9
num_columns = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses']
cat_columns = ['age_numeric', 'gender', 'race']

plt.figure(figsize=(10, 6))
plt.hist(df['num_medications'], bins=20, color='purple', edgecolor='black')
plt.title('Distribution of Number of Medications')
plt.xlabel('Number of Medications')
plt.ylabel('Frequency')
plt.show()

# Visualization 2: Readmission Rates by Age Group (Stacked Bar Plot)
if 'readmitted' in df.columns:
    df_age_readmit = df.groupby('age_numeric')['readmitted'].value_counts(normalize=True).unstack().fillna(0)
    df_age_readmit.plot(kind='bar', stacked=True, figsize=(12, 6), color=['skyblue', 'orange', 'green'])
    plt.title('Readmission Rates by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Proportion of Readmission')
    plt.legend(title='Readmission Status')
    plt.show()
