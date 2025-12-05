
import pandas as pd
from sklearn.model_selection import train_test_split

ori_df = pd.read_csv('Students Social Media Addiction.csv')

df = ori_df.copy()
df = df.drop(columns=["Student_ID"])

df = df.drop(columns=["Country", "Mental_Health_Score"])
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["Addicted_Score"])
y = df["Addicted_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

import joblib

joblib.dump(model, "rfreg.pkl")