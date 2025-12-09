import os
import matplotlib.pyplot as plt
from datetime import datetime

def salvar_fig(nome, pasta="reports/figures"):
    """
    Salva a figura atual do matplotlib com o nome especificado.
    """
    caminho = os.path.join(pasta, f"{nome}.png")
    plt.savefig(caminho, bbox_inches="tight")
    print(f"[OK] Figura salva em: {caminho}")


def salvar_texto(texto, nome_arquivo, pasta="reports/text"):
    """
    Salva um texto em um arquivo .txt.
    """
    caminho = os.path.join(pasta, nome_arquivo)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"[OK] Texto salvo em: {caminho}")


def salvar_texto_com_timestamp(texto, base_nome="analise"):
    """
    Salva texto com timestamp automático para versionamento.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{base_nome}_{timestamp}.txt"
    salvar_texto(texto, nome_arquivo)

