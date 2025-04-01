from model import model

if __name__ == "__main__":
    filepath = "titanic/train.csv"
    df = model.preprocess_data(filepath)
    y_test, y_pred = model.train_and_evaluate(df, model_type="random_forest")
    model.evaluate_metrics(y_test, y_pred)