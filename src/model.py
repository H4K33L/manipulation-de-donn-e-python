import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

class TitanicModel:
    def __init__(self, model_type="random_forest"):
        """ Initialise un modèle Random Forest ou Régression Logistique. """
        self.model = LogisticRegression(max_iter=1000) if model_type == "logistic" else RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """ Entraîne le modèle sur les données fournies. """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        
    def predict(self, X):
        """ Prédit les résultats à partir des données d'entrée. """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    @staticmethod
    def evaluate(y_test, y_pred):
        """ Évalue les performances du modèle. """
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Matrice de confusion :\n", cm)