import pandas as pd
from sklearn.impute import SimpleImputer

class TitanicPreprocessor:
    def __init__(self, imputer=None):
        """
        Initialise le préprocesseur par défaut avec la moyenne des valeurs.
        """
        self.imputer = imputer if imputer else SimpleImputer(strategy="mean")
    
    def fit(self, df):
        """
        Apprend les valeurs manquantes des colonnes Age et Fare.
        """
        self.imputer.fit(df[['Age', 'Fare']])
        return self
    
    def transform(self, df):
        """
        Applique les transformations : remplissage des valeurs manquantes, encodage et suppression de colonnes.
        """
        df = df.copy()
        df[['Age', 'Fare']] = self.imputer.transform(df[['Age', 'Fare']])
        df['isMale'] = df['Sex'].map({'male': 1, 'female': 0})
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin'], axis=1)
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
        return df
    
    def fit_transform(self, df):
        """
        Apprend et applique les transformations.
        """
        return self.fit(df).transform(df)