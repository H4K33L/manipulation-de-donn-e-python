import pandas as pd
from sklearn.impute import SimpleImputer

class TitanicPreprocessor:
    def __init__(self, imputer=None):
        """
        Initialise le préprocesseur avec un imputer facultatif.
        Si aucun imputer n'est fourni, un SimpleImputer avec une stratégie de moyenne est utilisé.
        """
        self.imputer = imputer if imputer else SimpleImputer(strategy="mean")

    def fit(self, df):
        """
        Apprend les statistiques nécessaires pour l'imputation sur les données fournies.
        :param df: DataFrame contenant les colonnes à imputer
        """
        self.age_median = df["Age"].median()
        self.fare_median = df['Fare'].median()
        self.most_frequent_embarkement = df['Embarked'].mode()[0]

    def transform(self, df):
        """
        Transforme les données en utilisant les statistiques apprises lors du fit.
        :param df: DataFrame à transformer
        :return: DataFrame prétraité
        """
        df['Age'] = df['Age'].fillna(self.age_median)
        df['Fare'] = df['Fare'].fillna(self.fare_median)
        df['Embarked'] = df['Embarked'].fillna(self.most_frequent_embarkement)
        df['isMale'] = df['Sex'].map({'male': 1, 'female': 0})
        df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin'], axis=1)
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

        return df

    def fit_transform(self, df):
        """
        Entraîne le préprocesseur et applique la transformation sur les données.
        :param df: DataFrame à traiter
        :return: DataFrame prétraité
        """
        self.fit(df)
        return self.transform(df)

    def preprocess_data(self, filepath):
        """
        Charge et prétraite les données à partir d'un fichier CSV.
        :param filepath: Chemin vers le fichier CSV
        :return: DataFrame prétraité
        """
        df = pd.read_csv(filepath)
        return self.fit_transform(df)