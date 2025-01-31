import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('heart.csv')

# Handle missing values if any (we can do this by filling with the median value)
df = df.fillna(df.median(), inplace=True)

# Encode categorical variables (example: Sex, ChestPainType)
df = pd.get_dummies(df, drop_first=True)

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('HeartDisease', axis=1))

# Choose target variable
y = df['HeartDisease']

# Feature selection - (Split the dataset into training and testing sets)
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.2, random_state=42)

# Use SMOTE for balancing the dataset
smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Save preprocessed data
pd.DataFrame(X_train_balanced).to_csv('X_train_balanced.csv', index=False)
pd.DataFrame(y_train_balanced).to_csv('y_train_balanced.csv', index=False)