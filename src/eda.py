import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import DATA_PATH
from src.features import criar_variaveis_derivadas
from src.utils_io import salvar_fig, salvar_texto

def carregar_dados():
    df = pd.read_csv(DATA_PATH)
    return df

def analise_churn(df):
    resumo = df['Attrition_Flag'].value_counts(normalize=True).rename("Proporção (%)") * 100
    print(resumo)

    # Gráfico de barras
    sns.countplot(data=df, x="Attrition_Flag")
    plt.title("Distribuição da Variável Attrition_Flag")
    plt.ylabel("Quantidade")
    plt.xlabel("Status do Cliente")
    salvar_fig("distribuicao_attrition_flag")
    plt.close()

    return resumo

def main():
    df = carregar_dados()
    df = criar_variaveis_derivadas(df)
    resumo = analise_churn(df)

    texto = "Resumo da variável target (churn):\n\n" + str(resumo)
    salvar_texto(texto, "resumo_churn.txt")

if __name__ == "__main__":
    main()

