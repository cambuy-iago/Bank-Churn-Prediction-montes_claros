# Perfeito! Aqui está um esboço de um `README.md` claro, técnico e amigável para seu projeto:

---

### 📘 Bank Churn Prediction - Projeto MBA

Este projeto tem como objetivo prever a evasão de clientes de cartão de crédito usando técnicas de aprendizado de máquina. Foi desenvolvido como projeto final do MBA em Ciência de Dados, com base em um case realista e estruturado com boas práticas de análise e engenharia de software.

---

### 📁 Estrutura do Repositório

```bash
Bank-Churn-Prediction/
├── data/                  # Base de dados original
├── notebooks/             # Notebooks com análises individuais
├── reports/
│   ├── figures/           # Gráficos salvos (png)
│   └── text/              # Relatórios de modelos (txt)
├── models/                # Modelos finais treinados (pkl)
├── src/                   # Scripts principais (config, features, modelagem)
├── webapp/                # Aplicação em Streamlit
└── requirements.txt       # Bibliotecas necessárias
```

---

### 🧠 Variáveis Derivadas Criadas

* `Ticket_Medio`, `Gasto_Medio_Mensal`, `Rotativo_Ratio`
* `Score_Relacionamento`, `Caiu_Valor`, `Caiu_Transacoes`
* `LTV_Proxy`, `Faixa_Idade`, `Renda_Class`, entre outras

---

### 📊 Modelos Treinados

1. XGBoost (`train_xgb.py`)
2. Random Forest (`train_rf.py`)
3. LightGBM (`train_lgbm.py`) ✅ Modelo final escolhido

---

### 🏆 Comparação de Modelos

| Modelo | AUC    | Métricas Gerais            |
| ------ | ------ | -------------------------- |
| LGBM   | 0.9826 | Excelente desempenho geral |
| XGB    | 0.9824 | Equilibrado e robusto      |
| RF     | 0.9770 | Bom, mas menos preciso     |

---

### 🖥️ WebApp - Previsão Interativa

Rode com:

```bash
streamlit run webapp/app.py
```

Interface simples para entrada de variáveis e retorno da probabilidade de churn.

---

### ✅ Como Executar

```bash
# Crie o ambiente
python -m venv .venv
.\.venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Execute notebooks ou o app
jupyter notebook
streamlit run webapp/app.py
```

---

### ✍️ Autoria

* **Autor:** Iago (MBA em Ciência de Dados - Montes Claros)
* **Data:** Dezembro 2025

---


