# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:39:39 2025
@author: fnegari

Association Rule Mining for Cleveland Heart Disease Dataset
"""

import pandas as pd
import numpy as np
from itertools import combinations

# Load Excel file
df = pd.read_excel(r"C:\Users\fnegari\Desktop\Cleveland_V1.xlsx")

# Set proper column names
df.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Drop missing values
df = df.dropna()

# --- Feature Engineering ---
df['Age_bin'] = (df['age'] >= 55).astype(int)
df['Sex_bin'] = df['sex'].astype(int)
df['cp_encoded'] = df['cp'].astype(int)
df['trestbps_bin'] = (df['trestbps'] > 120).astype(int)

df['chol_bin'] = pd.cut(df['chol'], bins=[0, 200, 240, float('inf')],
                        labels=[0, 1, 2]).astype(int)

df['thalach_level'] = pd.cut(df['thalach'], bins=[0, 100, 140, float('inf')],
                             labels=[0, 1, 2], include_lowest=True).astype(int)

df['fbs_bin'] = (df['fbs'] > 120).astype(int)
df['restecg_encoded'] = df['restecg'].astype(int)
df['exang_bin'] = df['exang'].astype(int)

df['oldpeak_encoded'] = pd.cut(df['oldpeak'], bins=[-0.1, 1.0, 2.0, df['oldpeak'].max()],
                               labels=[0, 1, 2]).astype(int)

df['slope_encoded'] = df['slope'].astype(int)
df['thal_encoded'] = df['thal'].astype(int)

df['disease'] = (df['target'] > 0).astype(int)

# Final DataFrame with selected features
df_final = df[[
    'Age_bin', 'Sex_bin', 'cp_encoded', 'trestbps_bin', 'chol_bin',
    'thalach_level', 'fbs_bin', 'restecg_encoded', 'exang_bin',
    'oldpeak_encoded', 'slope_encoded', 'ca', 'thal_encoded', 'disease'
]]

# Display sample
pd.set_option('display.max_columns', None)
print(df_final.head())

# --- Association Rule Mining (2 antecedents -> 1 consequent) ---
def pairwise_association_rules(df, target_col='disease'):
    results = []
    features = df.drop(columns=[target_col]).columns

    for f1, f2 in combinations(features, 2):
        for v1 in df[f1].dropna().unique():
            for v2 in df[f2].dropna().unique():
                condition = (df[f1] == v1) & (df[f2] == v2)
                subset = df[condition]
                support = len(subset) / len(df)
                if support == 0:
                    continue
                confidence = subset[target_col].mean()
                base_rate = df[target_col].mean()
                lift = confidence / base_rate if base_rate > 0 else np.nan
                results.append({
                    'Feature 1': f1,
                    'Value 1': v1,
                    'Feature 2': f2,
                    'Value 2': v2,
                    'Support': round(support, 4),
                    'Confidence': round(confidence, 4),
                    'Lift': round(lift, 4)
                })

    return pd.DataFrame(results).sort_values(by='Lift', ascending=False)

# Run association rule mining
rules_2to1 = pairwise_association_rules(df_final, target_col='disease')

# Save all rules
rules_2to1.to_csv("all_rules.csv", index=False)

# Save only strong rules
strong_rules = rules_2to1[rules_2to1['Lift'] > 1.2]
strong_rules.to_csv("strong_rules.csv", index=False)

# Print top 10 strong rules
print("Top Strong Association Rules (Lift > 1.2):")
print(strong_rules.head(10))
