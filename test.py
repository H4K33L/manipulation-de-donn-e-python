import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['isMale'] = df["Sex"].apply(lambda x: 1 if x == 'male' else 0)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    return df

def train_and_evaluate(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

if __name__ == "__main__":
    filepath = "train.csv"
    df = preprocess_data(filepath)
    train_and_evaluate(df)