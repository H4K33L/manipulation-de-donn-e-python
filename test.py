import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data(filepath, imputer=None):
    # tri des données, récuperation des importantes et supression des inutiles
    # mapage des donées non compréensible tel que sex
    df = pd.read_csv(filepath)
    if imputer is None:
        imputer = SimpleImputer(strategy="mean")
        df[['Age', 'Fare']] = imputer.fit_transform(df[['Age', 'Fare']])
    else:
        df[['Age', 'Fare']] = imputer.transform(df[['Age', 'Fare']])
    df['isMale'] = df["Sex"].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin'], axis=1)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    return df

def train_and_evaluate(df, model_type="random_forest"):
    # entrainement du modéle et comparaison avec les donées réelles
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=None, random_state=42)
    X_test = preprocess_data("titanic/test.csv")
    y_test = pd.read_csv("titanic/gender_submission.csv")['Survived']
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    return y_test, y_pred

def evaluate_metrics(y_test, y_pred):
    # évaluation de la présision
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Matrice de confusion :\n", cm)
    # Visualisation de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Sauvegarde de la matrice de confusion sous forme de PNG
    plt.savefig("confusion_matrix.png", format="png")
    plt.close()

# le lancement du programe
if __name__ == "__main__":
    filepath = "titanic/train.csv"
    df = preprocess_data(filepath)
    y_test, y_pred = train_and_evaluate(df, model_type="random_forest")
    evaluate_metrics(y_test, y_pred)
