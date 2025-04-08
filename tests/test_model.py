from src.model import TitanicModel

def test_train () :
    assert TitanicModel.train(), "le train a un éreur whaou !"

def test_predict (X) :
    assert len(TitanicModel.predict(X)) == X, "mauvais nobre de ligne"

def test_evaluate () :
    assert 0 <= TitanicModel.evaluate() <= 1, "metric incohérente"