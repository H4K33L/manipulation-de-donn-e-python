from src.model import Model
from src.preprocessor import Preprocessor

if __name__ == "__main__":
    filepath = "data/train.csv"
    preprocessor = Preprocessor()
    model = Model(model_type="random_forest")
    df = preprocessor.preprocess_data(filepath)
    y_test, y_pred = model.train_and_evaluate(df)
    model.evaluate_metrics(y_test, y_pred)