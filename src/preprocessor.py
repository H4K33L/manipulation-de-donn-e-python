import pandas as pd
from sklearn.impute import SimpleImputer

class Preprocessor:
    def __init__(self, imputer=None):
        """
        Initialise le préprocesseur avec un imputer facultatif.
        Si aucun imputer n'est fourni, un SimpleImputer avec une stratégie de moyenne est utilisé.
        """
        self.imputer = imputer if imputer else SimpleImputer(strategy="mean")

    def preprocess_data(self, filepath):
        """
        Charge et prétraite les données à partir d'un fichier CSV.
        Effectue le remplissage des valeurs manquantes, le mapping et la suppression des colonnes inutiles.
        
        :param filepath: Chemin vers le fichier CSV
        :return: DataFrame prétraité
        """
        df = pd.read_csv(filepath)
        df[['Age', 'Fare']] = self.imputer.fit_transform(df[['Age', 'Fare']])
        df['isMale'] = df['Sex'].map({'male': 1, 'female': 0})
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin'], axis=1)
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
        
        return df
