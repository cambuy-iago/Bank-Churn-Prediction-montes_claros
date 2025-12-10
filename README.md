# Bank-Churn-Prediction – Montes Claros

Sistema de **previsão de churn de clientes de cartão de crédito** usando Machine Learning,
desenvolvido como Projeto Aplicado do MBA em Ciência de Dados / Data Science & Analytics.

O objetivo é **identificar proativamente clientes com alto risco de evasão**, permitindo ações de
retenção mais eficientes e direcionadas.

---

## 🧩 1. Problema de Negócio

- Churn de clientes de cartão gera **perda de receita recorrente** e custos de aquisição desperdiçados.
- Reter um cliente geralmente é **5–7x mais barato** do que adquirir um novo.
- O banco precisa de um **score de risco de churn** por cliente para:
  - Priorizar campanhas de retenção
  - Estimar impacto financeiro
  - Monitorar a saúde da carteira

**Pergunta central:**  
> “Quais clientes têm maior probabilidade de encerrar o relacionamento nos próximos meses?”

---

## 📊 2. Base de Dados

- Dataset: `data/BankChurners.csv`
- Origem: base pública inspirada em um banco de varejo de cartões de crédito.
- Granularidade: **1 linha = 1 cliente**
- Principais grupos de variáveis:
  - Perfil demográfico: idade, dependentes, estado civil, escolaridade, renda
  - Relacionamento: tempo de casa, quantidade de produtos, contatos com o banco
  - Crédito: limite, saldo rotativo, utilização do limite
  - Transações: valor total, quantidade, variação entre trimestres (Q4 vs Q1)

Variável alvo:

- `Attrition_Flag` → transformada em `Attrition` (0 = cliente ativo, 1 = churn)

---

## 🧪 3. Metodologia

O projeto segue uma abordagem inspirada no **CRISP-DM**:

1. **Business Understanding**  
   - Entendimento do problema de churn e indicadores de sucesso (AUC, Recall da classe churn, impacto no negócio).

2. **Data Understanding & EDA**  
   - Análise exploratória (`notebooks/1_Analise_Exploratoria.ipynb` e `eda_completo.ipynb`)
   - PCA 2D/3D para inspeção de separabilidade
   - Clusterização para entender perfis de clientes e taxas de churn por cluster

3. **Data Preparation**  
   - Criação da base tratada: `data/base_tratada.csv`  
   - Criação da base de modelagem: `data/base_modelagem.csv`  
   - Feature engineering com variáveis de comportamento (ver abaixo)

4. **Modeling**  
   - Modelos avaliados:
     - Regressão Logística (baseline)
     - Random Forest
     - XGBoost
     - **LightGBM (modelo vencedor)**

5. **Evaluation**  
   - Métricas por modelo registradas em `reports/metrics_modelos.csv`
   - Relatórios de classificação em `reports/text/*.txt`
   - Curvas ROC, matrizes de confusão e importância de variáveis em `reports/figures/`

6. **Deployment / Uso**  
   - Script de pipeline (`src/pipeline_churn.py`)
   - Aplicação interactiva em Streamlit (`src/app_churn_streamlit.py`)

---

## 🧮 4. Feature Engineering

Principais variáveis derivadas criadas em `src/features.py`:

- **Ticket_Medio** – valor médio por transação  
- **Transacoes_por_Mes** – frequência de uso do cartão  
- **Gasto_Medio_Mensal** – intensidade de consumo mensal  
- **Rotativo_Ratio** – proporção do limite usada como saldo rotativo  
- **Disponibilidade_Relativa** – (limite – rotativo) / limite  
- **Caiu_Valor / Caiu_Transacoes** – flags de queda de gasto e de quantidade (Q4 vs Q1)  
- **Score_Relacionamento** – proxy de engajamento (quantidade de produtos)  
- **LTV_Proxy** – gasto médio mensal × meses de relacionamento  
- **Faixa_Idade, Renda_Class** – faixas categóricas para idade e renda

Estas features mostraram forte relação com o churn e foram fundamentais para o desempenho do modelo.

---

## 🤖 5. Modelagem e Resultados

### 5.1 Comparação de modelos (resumo)

Fonte: `reports/metrics_modelos.csv`

| Modelo                | Accuracy | ROC AUC | Precision (churn) | Recall (churn) | F1 (churn) |
|----------------------|---------:|--------:|-------------------:|---------------:|-----------:|
| Regressão Logística  | 0.853    | 0.920   | 0.528              | 0.815          | 0.641      |
| **LightGBM (final)** | **0.970**| **0.994**| **0.934**          | **0.874**      | **0.903**  |

- A Regressão Logística serve como baseline interpretável.
- O **LightGBM** apresentou:
  - **AUC ~ 0.99** (excelente capacidade de separação)
  - Alto **recall da classe churn**, importante para não perder clientes em risco
  - Robustez a desbalanceamento, com uso de `class_weight='balanced'` e 12 features selecionadas.

### 5.2 Análises de Interpretabilidade

Arquivos em `reports/figures/`:

- `shap_summary_plot.png` – impacto global das features no modelo LightGBM  
- `shap_bar_plot.png` – ranking de importância  
- `shap_dependence_Total_Trans_Ct.png` – relação entre nº de transações e risco de churn  
- `feature_importance_lightgbm.png` – importância de variáveis pelo modelo

Principais insights:

- Queda em **volume e valor de transações** é forte sinal de risco.
- Clientes com **poucos produtos** e **baixo relacionamento** têm maior probabilidade de churn.
- Padrões de uso do crédito (rotativo, utilização de limite) também contribuem significativamente.

---

## 💻 6. Arquitetura da Solução

**Pastas principais:**

```text
Bank-Churn-Prediction-montes_claros/
├── data/                 # Dados brutos e bases tratadas/modelagem
├── eda_results/          # Resultados consolidados de EDA
├── models/               # Modelos treinados (.pkl) e log de versões
├── notebooks/            # Notebooks Jupyter (EDA, modelagem, análises)
├── reports/
│   ├── figures/          # Gráficos (ROC, matriz de confusão, SHAP etc.)
│   └── text/             # Relatórios de métricas e classificação
├── src/
│   ├── 01_eda_base_tratada.py
│   ├── 02_model_training.py
│   ├── app_churn_streamlit.py
│   ├── features.py       # Feature engineering
│   ├── train_lgbm.py     # Treino LightGBM
│   ├── pipeline_churn.py # Orquestra o fluxo completo
│   └── ...               # Demais utilitários
└── webapp/
    └── app.py            # (versão alternativa / legado do app)
