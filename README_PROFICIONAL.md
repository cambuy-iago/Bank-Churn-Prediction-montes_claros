# Previsão de Churn Bancário — Banco Montes Claros

Projeto aplicado do MBA em Inteligência Artificial, Data Science & Big Data para Negócios (IBMEC), focado em **prever churn de clientes bancários** e apoiar a **estratégia de retenção** com base em Machine Learning interpretável.

> **Objetivo central:** antecipar quais clientes têm maior probabilidade de encerrar o relacionamento com o banco, permitindo ações proativas de retenção e priorização de contato.

---

## 🎯 Problema de Negócio

- Churn de clientes impacta diretamente **receita recorrente**, **custo de aquisição** e **rentabilidade de carteira**.
- O banco precisa de um mecanismo que:
  - identifique **clientes em risco**;
  - quantifique esse risco em termos de **probabilidade de churn**;
  - permita **simular cenários de atuação** (ex.: campanhas para segmentos específicos).

---

## 🗂️ Base de Dados

- Dataset derivado do **BankChurners** (domínio bancário, cartão de crédito).
- Cada linha representa um cliente com:
  - Perfil demográfico (idade, faixa de renda etc.)
  - Comportamento transacional (número de transações, valor médio, limite, utilização)
  - Relacionamento (tempo de relacionamento, número de produtos, tipo de cartão etc.)
- Target binária:
  - `Attrited Customer` → cliente que saiu
  - `Existing Customer` → cliente ativo  
  - Novo rótulo criado: **`Attrition`** (0 = permanece, 1 = churn)

---

## 🔬 Abordagem Analítica

1. **Entendimento de negócio e da base**
   - Mapeamento de variáveis com o time de negócio.
   - Discussão de hipóteses: quais comportamentos indicam risco de churn?

2. **Preparação e Feature Engineering**
   - Tratamento de nulos, outliers e balanceamento de classes.
   - Criação de variáveis derivadas:
     - intensidade de uso do cartão;
     - engajamento em canais;
     - proxies de rentabilidade.
   - Seleção de **conjunto enxuto de features** para facilitar deploy e explicabilidade.

3. **Modelagem supervisionada**
   - Modelos avaliados:
     - Regressão Logística
     - Random Forest
     - XGBoost
     - **LightGBM (modelo final)**
   - Métricas:
     - AUC-ROC
     - Recall da classe de churn
     - F1-score e matriz de confusão

4. **Interpretação e Explainability**
   - `feature_importance` nativa dos modelos em árvore.
   - SHAP para explicar:
     - impacto médio das variáveis;
     - casos individuais (por que este cliente está em risco?).

---

## 🧠 Modelo Final

- Algoritmo: **LightGBM Classifier**
- Justificativa:
  - Melhor equilíbrio entre **performance**, **tempo de treino** e **capacidade de generalização**.
  - Resultado robusto em AUC e métricas focadas em churn.
- Artefatos salvos na pasta `models/`:
  - `model_lgbm_v1.pkl` (modelo treinado)
  - `model_final.pkl` (modelo escolhido para produção)
  - `versions_log.csv` (histórico de versões)

---

## 📊 Principais Resultados (Visão de Banca)

- **AUC-ROC** consistente na base de teste (comparada entre modelos).
- Ganho expressivo de **recall de churners**, com controle de falsos positivos.
- Rankings de **variáveis mais importantes**:
  - intensidade de transações;
  - utilização de limite;
  - tempo de relacionamento;
  - número de produtos e interações.

Os gráficos e relatórios estão na pasta:

- `reports/figures/` → curvas ROC, matrizes de confusão, SHAP etc.
- `reports/text/` → métricas numéricas e relatórios em texto.

---

## 💻 Aplicativo Streamlit (Demo Executiva)

O app interativo foi desenvolvido em **Streamlit** para:

- Visualizar métricas e comparações de modelos;
- Fazer **predição individual** de clientes;
- Mostrar a probabilidade de churn com visualização tipo *gauge*.

> **Link do app (deploy Streamlit Cloud):**  
> _[inserir aqui a URL pública do app]_  

> **Arquivo principal do app:**  
> `src/app_churn_streamlit.py`

---

## 🧱 Estrutura do Repositório

```text
Bank-Churn-Prediction-montes_claros/
├── data/                  # Bases originais e tratadas
├── notebooks/             # EDA, modelagem, SHAP e comparações
├── src/                   # Código fonte (pipelines, treino, app)
├── models/                # Modelos treinados e controle de versões
├── reports/
│   ├── figures/           # Gráficos e visualizações
│   └── text/              # Métricas em texto
├── EXECUTIVE_SUMMARY.md   # Resumo de negócio para banca
├── IMPLEMENTATION_SUMMARY.md
├── README_PROFICIONAL.md  # Este arquivo
└── TESTING_GUIDE.md       # Guia de testes e replicação
