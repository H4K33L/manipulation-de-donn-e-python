import numpy as np
import pandas as pd
from src.model import TitanicModel
from sklearn.model_selection import train_test_split
data = {
    'Age': [22, 38, 26, 35, 28, 54, 2, 5, 27, 22],
    'Fare': [7.25, 71.2833, 7.925, 8.05, 8.45, 51.8625, 21.075, 24.45, 26.15, 10.05],
    'Embarked_C': [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
    'Embarked_S': [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    'isMale': [1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
}

df = pd.DataFrame(data)
X = df.drop(columns=['isMale'])
y = df['isMale']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = TitanicModel()

def test_train():
    try:
        model.train(X_train, y_train)
        assert hasattr(model, 'model'), "Erreur: le modèle n'a pas été entraîné correctement"
        print("Le test train() a réussi.")
    except Exception as e:
        print(f"Le test train() a échoué: {e}")

def test_predict():
    try:
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test), f"Erreur: le nombre de lignes dans les prédictions ({len(predictions)}) ne correspond pas à X_test ({len(X_test)})"
        print("Le test predict() a réussi.")
    except Exception as e:
        print(f"Le test predict() a échoué: {e}")

def test_evaluate():
    try:
        y_pred = model.predict(X_test)
        accuracy = TitanicModel.evaluate(y_test, y_pred)
        assert 0 <= accuracy <= 1, f"Erreur: l'accuracy devrait être entre 0 et 1, mais il est {accuracy}"
        print("Le test evaluate() a réussi.")
    except Exception as e:
        print(f"Le test evaluate() a échoué: {e}")
