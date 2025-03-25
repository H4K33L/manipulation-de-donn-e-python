import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['isMale'] = df["Sex"].map({'male': 1, 'female': 0})
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin'], axis=1)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    return df

def train_and_evaluate(df, model_type="logistic"):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred

def evaluate_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Matrice de confusion :\n", cm)

if __name__ == "__main__":
    filepath = "train.csv"
    df = preprocess_data(filepath)
    y_test, y_pred = train_and_evaluate(df, model_type="random_forest")
    evaluate_metrics(y_test, y_pred)
