import pandas as pd
from sklearn.impute import SimpleImputer

class preprocessor() :
    def preprocess_data(filepath, imputer=None):
        # tri des données, récuperation des importantes et supresion des inutiles
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