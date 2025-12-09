import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

def treinar_modelo(modelo, X_train, y_train):
    modelo.fit(X_train, y_train)
    return modelo

def obter_modelo(nome):
    if nome == "xgb":
        return xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    elif nome == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    elif nome == "lgbm":
        return lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    else:
        raise ValueError(f"Modelo '{nome}' não reconhecido.")
