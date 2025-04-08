import pandas as pd
import pytest
from src.preprocessor import TitanicPreprocessor

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
    return pd.DataFrame(data)

@pytest.fixture
def preprocessor():
    return TitanicPreprocessor()

def test_fit(preprocessor, datasets):
    preprocessor.fit(datasets)
    assert preprocessor.age_median == 34.75, f"Erreur: la médiane de l'âge attendue est 34.75, mais obtenue {preprocessor.age_median}"
    assert preprocessor.fare_median == 9.275, f"Erreur: la médiane de la Fare attendue est 9.275, mais obtenue {preprocessor.fare_median}"
    assert preprocessor.most_frequent_embarkement == 'S', f"Erreur: l'embarquement le plus fréquent attendu est 'S', mais obtenu {preprocessor.most_frequent_embarkement}"
    print("Le test fit() a réussi.")

def test_transform(preprocessor, datasets):
    preprocessor.fit(datasets)
    df_transformed = preprocessor.transform(datasets)
    assert df_transformed['Age'].isna().sum() == 0, "Erreur: des valeurs manquantes existent encore dans la colonne 'Age'"
    assert df_transformed['Fare'].isna().sum() == 0, "Erreur: des valeurs manquantes existent encore dans la colonne 'Fare'"
    assert 'Name' not in df_transformed.columns, "Erreur: la colonne 'Name' n'a pas été supprimée"
    assert 'Sex' not in df_transformed.columns, "Erreur: la colonne 'Sex' n'a pas été supprimée"
    assert 'Ticket' not in df_transformed.columns, "Erreur: la colonne 'Ticket' n'a pas été supprimée"
    assert 'Cabin' not in df_transformed.columns, "Erreur: la colonne 'Cabin' n'a pas été supprimée"
    assert 'isMale' in df_transformed.columns, "Erreur: la colonne 'isMale' n'a pas été créée"
    assert df_transformed['isMale'].iloc[0] == 1, "Erreur: la valeur de 'isMale' pour le premier enregistrement devrait être 1 (male)"
    assert df_transformed['isMale'].iloc[1] == 0, "Erreur: la valeur de 'isMale' pour le deuxième enregistrement devrait être 0 (female)"
    assert 'Embarked' not in df_transformed.columns, "Erreur: la colonne 'Embarked' n'a pas été supprimée"
    print("Le test transform() a réussi.")

def test_fit_transform(preprocessor, datasets):
    df_fit_transformed = preprocessor.fit_transform(datasets)
    df_transformed = preprocessor.transform(datasets)
    assert df_fit_transformed.shape == df_transformed.shape, "Erreur: les formes des DataFrames après fit_transform et transform ne correspondent pas"
    assert df_fit_transformed.equals(df_transformed), "Erreur: les DataFrames après fit_transform et transform ne sont pas identiques"
    print("Le test fit_transform() a réussi.")

