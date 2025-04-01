import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.preprocessor import Preprocessor

class Model:
    def __init__(self, model_type="random_forest"):
        """
        Initialise le modèle avec le type spécifié (random_forest ou logistic).
        """
        if model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.scaler = StandardScaler()
        self.preprocessor = Preprocessor()
    
    def train_and_evaluate(self, df):
        """
        Entraîne le modèle sur les données fournies et l'évalue sur un ensemble de test.
        
        :param df: DataFrame contenant les données d'entraînement
        :return: Tuple (y_test, y_pred)
        """
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        return y_test, y_pred
    
    @staticmethod
    def evaluate_metrics(y_test, y_pred):
        """
        Évalue les performances du modèle avec des métriques d'exactitude et une matrice de confusion.
        
        :param y_test: Valeurs réelles
        :param y_pred: Valeurs prédites
        """
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print("Matrice de confusion :\n", cm)