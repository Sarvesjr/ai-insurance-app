import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/insurance_data.csv")
df = df.drop(['id'], axis=1)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
df['Vehicle_Age'] = df['Vehicle_Age'].map({
    '< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2
})
X = df.drop('Response', axis=1)
y = df['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
joblib.dump(model, "model/claim_model.pkl")
print("Model saved ✅")