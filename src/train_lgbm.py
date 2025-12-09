import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import criar_variaveis_derivadas
from src.utils_io import salvar_fig, salvar_texto
from src.config import DATA_PATH
from src.train_model import obter_modelo

def carregar_e_preparar_dados():
    df = pd.read_csv(DATA_PATH)
    df = criar_variaveis_derivadas(df)
    y = df["Attrition_Flag"].map({"Attrited Customer": 1, "Existing Customer": 0})
    X = df[[
        'Customer_Age', 'Dependent_count', 'Credit_Limit',
        'Total_Trans_Amt', 'Total_Trans_Ct', 'Ticket_Medio',
        'Gasto_Medio_Mensal', 'Rotativo_Ratio', 'Score_Relacionamento',
        'LTV_Proxy', 'Caiu_Valor', 'Caiu_Transacoes'
    ]]
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def avaliar_modelo(y_test, y_pred, y_proba, nome_prefixo):
    report = classification_report(y_test, y_pred, digits=4)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    salvar_texto(report, f"relatorio_{nome_prefixo}.txt")
    salvar_texto(f"AUC: {auc:.4f}", f"auc_{nome_prefixo}.txt")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão - {nome_prefixo}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    salvar_fig(f"matriz_confusao_{nome_prefixo}")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], "k--")
    plt.title(f"Curva ROC - {nome_prefixo}")
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.legend()
    salvar_fig(f"curva_roc_{nome_prefixo}")
    plt.close()

def main():
    X_train, X_test, y_train, y_test = carregar_e_preparar_dados()
    modelo = obter_modelo("lgbm")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:,1]
    avaliar_modelo(y_test, y_pred, y_proba, "lgbm")

if __name__ == "__main__":
    main()
