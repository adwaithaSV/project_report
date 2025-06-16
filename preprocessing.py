import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
file_path = 'gene_expression.csv'
df = pd.read_csv(file_path)
print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print("Missing values:\n", df.isnull().sum())
df.dropna(inplace=True)  # or df.fillna(method='ffill', inplace=True)
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
#  Final success message
print("Data preprocessing successful ")
