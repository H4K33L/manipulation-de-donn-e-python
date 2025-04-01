import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessor import TitanicPreprocessor
from src.model import TitanicModel

if __name__ == "__main__":
    filepath = "data/train.csv"
    preprocessor = TitanicPreprocessor()
    model = TitanicModel(model_type="random_forest")
    
    df = preprocessor.fit_transform(pd.read_csv(filepath))
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    
    model.evaluate(y_test, y_pred)
