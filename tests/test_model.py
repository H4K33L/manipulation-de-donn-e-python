import pandas as pd
import pytest
from src.model import TitanicModel
from sklearn.model_selection import train_test_split

@pytest.fixture
def datasets():
    data = {
        'PassengerId': [892, 893, 894, 895, 896, 897, 898, 899, 900, 901],
        'Pclass': [3, 3, 1, 1, 3, 2, 2, 3, 1, 3],
        'Name': [
            "Kelly, Mr. James", "Wilkes, Mrs. James (Ellen Needs)", "Myles, Mr. Richard", "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)", 
            "Nasser, Mrs. Nicholas (Adele Achem)", "Sandstrom, Miss. Marguerite Rut", "Bourke, Mr. John", "Strom, Mr. Thomas", 
            "Krause, Mr. Victor", "Harris, Mrs. William (Margaret Waugh)"
        ],
        'Sex': ['male', 'female', 'male', 'female', 'female', 'female', 'male', 'male', 'male', 'female'],
        'Age': [34.5, 47, 62, 35, 27, 14, 41, 28, 34, 40],
        'SibSp': [0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
        'Parch': [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        'Ticket': ['330911', '363272', '123456', '112233', '456789', '234567', '345678', '987654', '567890', '678901'],
        'Fare': [7.8292, 7.0, 71.2833, 7.25, 8.5, 14.4542, 21.075, 8.05, 10.05, 10.5],
        'Cabin': [None, None, 'C85', 'C123', None, 'E46', None, None, None, 'B28'],
        'Embarked': ['Q', 'S', 'S', 'C', 'C', 'S', 'S', 'S', 'S', 'S'],
        'Survived': [0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    return train_test_split(X, y, test_size=0.3, random_state=42)

@pytest.fixture
def model():
    return TitanicModel()

def test_train(datasets,model):
    try:
        X_train,y_train=datasets[0],datasets[1]
        model.train(X_train, y_train)
        assert hasattr(model, 'model'), "Erreur: le modèle n'a pas été entraîné correctement"
        print("Le test train() a réussi.")
    except Exception as e:
        print(f"Le test train() a échoué: {e}")

def test_predict(datasets,model):
    try:
        X_test=datasets[2]
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test), f"Erreur: le nombre de lignes dans les prédictions ({len(predictions)}) ne correspond pas à X_test ({len(X_test)})"
        print("Le test predict() a réussi.")
    except Exception as e:
        print(f"Le test predict() a échoué: {e}")

def test_evaluate(datasets,model):
    try:
        X_test,y_test=datasets[2],datasets[3]
        y_pred = model.predict(X_test)
        accuracy = TitanicModel.evaluate(y_test, y_pred)
        assert 0 <= accuracy <= 1, f"Erreur: l'accuracy devrait être entre 0 et 1, mais il est {accuracy}"
        print("Le test evaluate() a réussi.")
    except Exception as e:
        print(f"Le test evaluate() a échoué: {e}")
