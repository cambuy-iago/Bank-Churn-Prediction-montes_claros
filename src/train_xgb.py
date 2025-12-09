import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

from src.features import criar_variaveis_derivadas
from src.utils_io import salvar_fig, salvar_texto
from src.config import DATA_PATH

def carregar_e_preparar_dados():
    df = pd.read_csv(DATA_PATH)
    df = criar_variaveis_derivadas(df)

    y = df["Attrition_Flag"].map({
        "Attrited Customer": 1,
        "Existing Customer": 0
    })

    X = df[[
        'Customer_Age', 'Dependent_count', 'Credit_Limit',
        'Total_Trans_Amt', 'Total_Trans_Ct', 'Ticket_Medio',
        'Gasto_Medio_Mensal', 'Rotativo_Ratio', 'Score_Relacionamento',
        'LTV_Proxy', 'Caiu_Valor', 'Caiu_Transacoes'
    ]]

    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def avaliar_modelo(y_test, y_pred, y_proba):
    report = classification_report(y_test, y_pred, digits=4)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    salvar_texto(report, "relatorio_xgb.txt")
    salvar_texto(f"AUC: {auc:.4f}", "auc_xgb.txt")

    # Matriz de confusão
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão - XGBoost")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    salvar_fig("matriz_confusao_xgb")
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], "k--")
    plt.title("Curva ROC - XGBoost")
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.legend()
    salvar_fig("curva_roc_xgb")
    plt.close()

def main():
    X_train, X_test, y_train, y_test = carregar_e_preparar_dados()

    modelo = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:,1]

    avaliar_modelo(y_test, y_pred, y_proba)

if __name__ == "__main__":
    main()