from pathlib import Path
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------
# CONFIGURAÇÃO DE CAMINHOS COM FALLBACKS ROBUSTOS
# -----------------------------------------------------------
def setup_paths():
    """Configura os caminhos do projeto com múltiplos fallbacks"""
    
    # Tenta encontrar a raiz do projeto de diferentes maneiras
    current_file = Path(__file__).resolve()
    
    # Opção 1: Se o app está em src/
    project_root = current_file.parent.parent
    
    # Verifica se a estrutura está correta
    if not (project_root / "data").exists():
        # Opção 2: Tenta um nível acima
        project_root = current_file.parent.parent.parent
    
    # Fallback: Caminho absoluto baseado na sua estrutura
    if not (project_root / "data").exists():
        fallback_path = Path(r"C:\Users\Iago\OneDrive\Desktop\Projeto Churn\Bank-Churn-Prediction-montes_claros")
        if fallback_path.exists():
            project_root = fallback_path
    
    # Caminhos principais
    MODEL_PATH = project_root / "models" / "model_final.pkl"
    SCALER_PATH = project_root / "models" / "scaler.pkl"
    METRICS_PATH = project_root / "reports" / "metrics_modelos.csv"
    FIG_CM_PATH = project_root / "reports" / "figures" / "matriz_confusao_lightgbm.png"
    FIG_ROC_PATH = project_root / "reports" / "figures" / "roc_curve_lightgbm.png"
    DATA_PATH = project_root / "data" / "BankChurners.csv"
    
    # Adiciona src ao sys.path para importações
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.append(str(src_path))
    
    return {
        "PROJECT_ROOT": project_root,
        "MODEL_PATH": MODEL_PATH,
        "SCALER_PATH": SCALER_PATH,
        "METRICS_PATH": METRICS_PATH,
        "FIG_CM_PATH": FIG_CM_PATH,
        "FIG_ROC_PATH": FIG_ROC_PATH,
        "DATA_PATH": DATA_PATH
    }

# Obter caminhos configurados
paths = setup_paths()
PROJECT_ROOT = paths["PROJECT_ROOT"]
MODEL_PATH = paths["MODEL_PATH"]
SCALER_PATH = paths["SCALER_PATH"]
METRICS_PATH = paths["METRICS_PATH"]
FIG_CM_PATH = paths["FIG_CM_PATH"]
FIG_ROC_PATH = paths["FIG_ROC_PATH"]
DATA_PATH = paths["DATA_PATH"]

# -----------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# -----------------------------------------------------------
st.set_page_config(
    page_title="Banco Mercantil - Preditor de Churn",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar visual
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# CARREGAMENTO DE MODELO E SCALER
# -----------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    """Carrega o modelo e o scaler com fallbacks robustos"""
    try:
        # Carregar modelo
        if MODEL_PATH.exists():
            modelo = joblib.load(MODEL_PATH)
            st.sidebar.success("✅ Modelo carregado com sucesso")
        else:
            st.sidebar.error(f"❌ Modelo não encontrado em: {MODEL_PATH}")
            st.sidebar.info("💡 Execute o script de treinamento primeiro")
            return None, None
        
        # Carregar scaler se existir
        scaler = None
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            st.sidebar.success("✅ Scaler carregado com sucesso")
        
        return modelo, scaler
        
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return None, None

modelo, scaler = load_model_and_scaler()

# -----------------------------------------------------------
# FUNÇÕES DE FEATURE ENGINEERING (FALLBACK SE src.features NÃO DISPONÍVEL)
# -----------------------------------------------------------
def criar_variaveis_derivadas_fallback(df):
    """
    Função de fallback para criar variáveis derivadas se o módulo src.features não estiver disponível
    """
    df = df.copy()
    
    # 1. Features básicas com tratamento de divisão por zero
    df["Ticket_Medio"] = np.where(df["Total_Trans_Ct"] != 0, 
                                  df["Total_Trans_Amt"] / df["Total_Trans_Ct"], 
                                  0)
    
    df["Transacoes_por_Mes"] = np.where(df["Months_on_book"] != 0, 
                                        df["Total_Trans_Ct"] / df["Months_on_book"], 
                                        0)
    
    df["Gasto_Medio_Mensal"] = np.where(df["Months_on_book"] != 0, 
                                        df["Total_Trans_Amt"] / df["Months_on_book"], 
                                        0)
    
    # 2. Utilização de crédito
    df["Rotativo_Ratio"] = np.where(df["Credit_Limit"] != 0, 
                                    df["Total_Revolving_Bal"] / df["Credit_Limit"], 
                                    0)
    
    df["Disponibilidade_Relativa"] = np.where(df["Credit_Limit"] != 0, 
                                              (df["Credit_Limit"] - df["Total_Revolving_Bal"]) / df["Credit_Limit"], 
                                              0)
    
    # 3. Flags de variação
    df["Caiu_Transacoes"] = (df["Total_Ct_Chng_Q4_Q1"] < 1).astype(int)
    df["Caiu_Valor"] = (df["Total_Amt_Chng_Q4_Q1"] < 1).astype(int)
    
    # 4. Relacionamento
    df["Score_Relacionamento"] = df["Total_Relationship_Count"]
    df["LTV_Proxy"] = df["Gasto_Medio_Mensal"] * df["Months_on_book"]
    
    # 5. Faixa etária
    def faixa_idade(x):
        if x < 30:
            return "<30"
        elif x < 50:
            return "30-49"
        elif x < 70:
            return "50-69"
        else:
            return "70+"
    
    df["Faixa_Idade"] = df["Customer_Age"].apply(faixa_idade)
    
    # 6. Classificação de renda
    def renda_class(ic):
        if ic in ["$60K - $80K", "$80K - $120K", "$120K +"]:
            return "Alta"
        elif ic in ["$40K - $60K", "$20K - $40K"]:
            return "Média"
        else:
            return "Baixa"
    
    df["Renda_Class"] = df["Income_Category"].apply(renda_class)
    
    # 7. Criar flag de churn se a coluna existir
    if "Attrition_Flag" in df.columns:
        df["churn_flag"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
    
    return df

# Tenta importar a função original, usa fallback se falhar
try:
    from src.features import criar_variaveis_derivadas
    criar_variaveis_derivadas_wrapper = criar_variaveis_derivadas
except ImportError:
    st.sidebar.warning("⚠️ Usando função de fallback para criar_variáveis_derivadas")
    criar_variaveis_derivadas_wrapper = criar_variaveis_derivadas_fallback

# -----------------------------------------------------------
# CARREGAMENTO DE DADOS
# -----------------------------------------------------------
@st.cache_data
def load_data_raw():
    """Carrega os dados brutos com múltiplos fallbacks"""
    # Lista de possíveis caminhos
    possible_paths = [
        DATA_PATH,
        Path("data/BankChurners.csv"),
        Path(r"C:\Users\Iago\OneDrive\Desktop\Projeto Churn\Bank-Churn-Prediction-montes_claros\data\BankChurners.csv"),
        PROJECT_ROOT / "BankChurners.csv"
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Dados carregados de: {path}")
                return df
            except Exception as e:
                continue
    
    st.sidebar.error("❌ Não foi possível carregar os dados. Verifique o caminho do arquivo.")
    return None

@st.cache_data
def load_data_with_features():
    """Carrega os dados e aplica feature engineering"""
    df = load_data_raw()
    if df is None:
        return None
    
    # Aplica feature engineering
    df = criar_variaveis_derivadas_wrapper(df)
    return df

# -----------------------------------------------------------
# DICIONÁRIOS DE TRADUÇÃO (ATUALIZADOS)
# -----------------------------------------------------------
DIC_NOME_PT_NUMERICOS = {
    "Idade do Cliente": "Customer_Age",
    "Número de Dependentes": "Dependent_count",
    "Meses de Relacionamento": "Months_on_book",
    "Quantidade de Produtos com o Banco": "Total_Relationship_Count",
    "Meses Inativo (12 meses)": "Months_Inactive_12_mon",
    "Contatos com o Banco (12 meses)": "Contacts_Count_12_mon",
    "Limite de Crédito": "Credit_Limit",
    "Saldo Rotativo": "Total_Revolving_Bal",
    "Variação de Valor Q4/Q1": "Total_Amt_Chng_Q4_Q1",
    "Valor Total Transacionado (12 meses)": "Total_Trans_Amt",
    "Número de Transações (12 meses)": "Total_Trans_Ct",
    "Variação de Transações Q4/Q1": "Total_Ct_Chng_Q4_Q1",
    "Utilização Média do Limite": "Avg_Utilization_Ratio",
    "Score de Relacionamento": "Score_Relacionamento",
    "Proxy LTV": "LTV_Proxy",
    "Caiu em Valor": "Caiu_Valor",
    "Caiu em Transações": "Caiu_Transacoes",
}

DIC_NOME_PT_ENGINEERED = {
    "Ticket Médio por Transação": "Ticket_Medio",
    "Transações por Mês": "Transacoes_por_Mes",
    "Gasto Médio Mensal": "Gasto_Medio_Mensal",
    "Uso do Rotativo (Ratio)": "Rotativo_Ratio",
    "Disponibilidade Relativa de Limite": "Disponibilidade_Relativa",
    "Faixa de Idade": "Faixa_Idade",
    "Classificação de Renda": "Renda_Class",
}

# -----------------------------------------------------------
# FUNÇÕES AUXILIARES PARA PREVISÃO
# -----------------------------------------------------------
def calcular_features_engineered_row(row: dict) -> dict:
    """Calcula todas as features derivadas para uma única linha"""
    # Valores básicos com proteção contra divisão por zero
    idade = row.get("Customer_Age", 0)
    months_on_book = max(row.get("Months_on_book", 1), 1)
    credit_limit = max(row.get("Credit_Limit", 1.0), 0.1)
    total_trans_amt = row.get("Total_Trans_Amt", 0)
    total_trans_ct = max(row.get("Total_Trans_Ct", 1), 1)
    total_revolving_bal = row.get("Total_Revolving_Bal", 0)
    total_relationship_count = row.get("Total_Relationship_Count", 0)
    total_amt_chng_q4_q1 = row.get("Total_Amt_Chng_Q4_Q1", 1.0)
    total_ct_chng_q4_q1 = row.get("Total_Ct_Chng_Q4_Q1", 1.0)
    
    # Cálculo das features
    ticket_medio = total_trans_amt / total_trans_ct if total_trans_ct > 0 else 0
    transacoes_mes = total_trans_ct / months_on_book if months_on_book > 0 else 0
    gasto_mensal = total_trans_amt / months_on_book if months_on_book > 0 else 0
    rotativo_ratio = total_revolving_bal / credit_limit if credit_limit > 0 else 0
    disponibilidade_relativa = (credit_limit - total_revolving_bal) / credit_limit if credit_limit > 0 else 0
    
    # Faixa etária
    if idade < 30:
        faixa_idade = "<30"
    elif idade < 50:
        faixa_idade = "30-49"
    elif idade < 70:
        faixa_idade = "50-69"
    else:
        faixa_idade = "70+"
    
    # Classificação de renda
    income = row.get("Income_Category", "")
    if income in ["$60K - $80K", "$80K - $120K", "$120K +"]:
        renda_class = "Alta"
    elif income in ["$40K - $60K", "$20K - $40K"]:
        renda_class = "Média"
    else:
        renda_class = "Baixa"
    
    # Score de relacionamento e LTV Proxy
    score_relacionamento = total_relationship_count
    ltv_proxy = gasto_mensal * months_on_book
    
    # Flags de queda
    caiu_valor = 1 if total_amt_chng_q4_q1 < 1 else 0
    caiu_transacoes = 1 if total_ct_chng_q4_q1 < 1 else 0
    
    # Atualiza o dicionário com todas as features
    row.update({
        "Ticket_Medio": ticket_medio,
        "Transacoes_por_Mes": transacoes_mes,
        "Gasto_Medio_Mensal": gasto_mensal,
        "Rotativo_Ratio": rotativo_ratio,
        "Disponibilidade_Relativa": disponibilidade_relativa,
        "Faixa_Idade": faixa_idade,
        "Renda_Class": renda_class,
        "Score_Relacionamento": score_relacionamento,
        "LTV_Proxy": ltv_proxy,
        "Caiu_Valor": caiu_valor,
        "Caiu_Transacoes": caiu_transacoes,
    })
    
    return row

def montar_dataframe_previsao(row: dict) -> pd.DataFrame:
    colunas_numericas = [
        "Customer_Age", "Dependent_count", "Months_on_book",
        "Total_Relationship_Count", "Months_Inactive_12_mon",
        "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
        "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
        "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
        "Ticket_Medio", "Transacoes_por_Mes", "Gasto_Medio_Mensal",
        "Rotativo_Ratio", "Disponibilidade_Relativa",
    ]

    colunas_categoricas = [
        "Gender", "Education_Level", "Marital_Status",
        "Income_Category", "Card_Category", "Faixa_Idade", "Renda_Class",
    ]

    colunas = colunas_numericas + colunas_categoricas
    for col in colunas:
        if col not in row:
            row[col] = None

    df = pd.DataFrame([row], columns=colunas)
    return df

def prever_cliente(row: dict) -> tuple[float, int]:
    if modelo is None:
        return 0.0, 0
    
    row_eng = calcular_features_engineered_row(row)
    df = montar_dataframe_previsao(row_eng)
    prob = float(modelo.predict_proba(df)[0][1])
    classe = int(modelo.predict(df)[0])
    return prob, classe

def criar_gauge_chart(valor, titulo):
    """Cria um gráfico gauge para visualização de probabilidade"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor * 100,
        title={'text': titulo, 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
st.sidebar.title("💳 Preditor de Churn")
st.sidebar.markdown("**MBA – Projeto Aplicado**")
st.sidebar.markdown("---")

aba = st.sidebar.radio(
    "📱 Navegação:",
    [
        "🏠 Início",
        "📈 Visão Geral do Modelo",
        "📊 Análise Exploratória",
        "👥 Exemplos Práticos",
        "👤 Simulador Individual",
        "📂 Análise em Lote",
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Dica de Navegação:**
- Comece pelo **Início** para entender o contexto
- Explore os **Exemplos Práticos** para ver casos reais
- Use o **Simulador** para testar cenários
""")

# -----------------------------------------------------------
# ABA 0 – INÍCIO
# -----------------------------------------------------------
if aba.startswith("🏠"):
    st.markdown('<div class="main-header">🏦 Sistema de Predição de Churn Bancário</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 👋 Bem-vindo ao Sistema de Previsão de Evasão de Clientes
    
    Este sistema utiliza **Inteligência Artificial** para identificar clientes com alta probabilidade 
    de deixar o banco, permitindo ações preventivas de retenção.
    """)
    
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=100, caption="Banco Mercantil")
    
    st.markdown("---")
    
    st.subheader("📊 Visão Geral do Projeto")
    st.markdown("""
    Este projeto tem como objetivo prever a probabilidade de um cliente deixar o banco (churn) utilizando 
    técnicas de machine learning. Abaixo estão algumas seções importantes do sistema:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. Visão Geral do Modelo")
        st.markdown("""
        - Entenda como o modelo foi construído e avaliado.
        - Visualize métricas de desempenho como AUC-ROC e matriz de confusão.
        """)
        
        st.markdown("#### 2. Análise Exploratória")
        st.markdown("""
        - Explore os dados utilizados no treinamento do modelo.
        - Visualize a distribuição de variáveis e a relação com o churn.
        """)
    
    with col2:
        st.markdown("#### 3. Exemplos Práticos")
        st.markdown("""
        - Veja exemplos reais de previsões do modelo.
        - Entenda como interpretar as saídas do sistema.
        """)
        
        st.markdown("#### 4. Simulador Individual")
        st.markdown("""
        - Simule a probabilidade de churn para um cliente específico.
        - Ajuste variáveis e veja o impacto na previsão.
        """)
        
        st.markdown("#### 5. Análise em Lote")
        st.markdown("""
        - Faça upload de uma lista de clientes e obtenha previsões em massa.
        - Receba um relatório detalhado com insights sobre cada cliente.
        """)
    
    st.markdown("---")
    
    st.subheader("📈 Próximos Passos")
    st.markdown("""
    1. **Explore a aba "📊 Análise Exploratória"** para entender os dados.
    2. **Veja os "👥 Exemplos Práticos"** para entender saídas do modelo.
    3. **Use o "👤 Simulador Individual"** para testar cenários específicos.
    4. **Realize uma "📂 Análise em Lote"** para previsões em massa.
    """)
    
    st.markdown("---")
    
    st.subheader("ℹ️ Informações Adicionais")
    st.markdown("""
    - Este projeto é parte de um trabalho acadêmico do MBA em Data Science.
    - Para mais informações, entre em contato com o desenvolvedor.
    """)
    
    st.markdown("---")
    
    st.subheader("🔧 Configurações Avançadas")
    with st.expander("Clique aqui para opções avançadas", expanded=False):
        st.markdown("""
        - **Recarregar Modelo:** Force o recarregamento do modelo e scaler.
        - **Caminhos de Dados:** Veja e edite os caminhos dos arquivos de dados.
        """)
        
        if st.button("🔄 Recarregar Modelo"):
            modelo, scaler = load_model_and_scaler()
        
        st.markdown("**Caminhos Atuais:**")
        st.markdown(f"- Modelo: `{MODEL_PATH}`")
        st.markdown(f"- Scaler: `{SCALER_PATH}`")
        st.markdown(f"- Dados: `{DATA_PATH}`")
        
        if st.button("📂 Abrir Pasta do Projeto"):
            os.startfile(PROJECT_ROOT)

# -----------------------------------------------------------
# ABA 1 – VISÃO GERAL DO MODELO
# -----------------------------------------------------------
if aba.startswith("📈"):
    st.markdown('<div class="main-header">📈 Visão Geral do Modelo</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Nesta seção, você pode entender como o modelo foi construído e avaliado. 
    As informações abaixo mostram as principais métricas de desempenho do modelo.
    """)
    
    # Exibir métricas do modelo
    @st.cache_data
    def carregar_metricas():
        if not METRICS_PATH.exists():
            st.error("Arquivo de métricas não encontrado!")
            return None
        return pd.read_csv(METRICS_PATH)
    
    df_metricas = carregar_metricas()
    
    if df_metricas is not None:
        # Fix for ValueError: Invalid column name 'Métrica'
        fig_metricas = px.bar(
            df_metricas,
            x='modelo',  # Corrected column name
            y='roc_auc_mean',
            error_y='roc_auc_std',
            title='Métricas por Modelo',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_metricas, use_container_width=True, key='fig_metricas')  # Added unique key
    
    st.markdown("---")
    
    st.subheader("📊 Matriz de Confusão")
    st.markdown("""
    A matriz de confusão abaixo mostra o desempenho do modelo em termos de verdadeiros positivos, 
    falsos positivos, verdadeiros negativos e falsos negativos.
    """)
    
    if FIG_CM_PATH.exists():
        st.image(str(FIG_CM_PATH), caption="Matriz de Confusão", use_container_width=True)
    else:
        st.warning("Matriz de confusão não disponível.")
    
    st.markdown("---")
    
    st.subheader("📈 Curva ROC")
    st.markdown("""
    A curva ROC (Receiver Operating Characteristic) ilustra a capacidade do modelo em classificar 
    corretamente os casos positivos e negativos. Abaixo está a curva ROC do modelo.
    """)
    
    if FIG_ROC_PATH.exists():
        st.image(str(FIG_ROC_PATH), caption="Curva ROC", use_column_width=True)
    else:
        st.warning("Curva ROC não disponível.")
    
    st.markdown("---")
    
    st.subheader("🔍 Interpretação do Modelo")
    st.markdown("""
    O gráfico abaixo mostra a importância das variáveis utilizadas pelo modelo para fazer previsões. 
    Variáveis com maior importância têm mais impacto na decisão do modelo.
    """)
    
    # Gráfico de importância das variáveis
    if modelo is not None:
        importancia = modelo.feature_importances_
        nomes_variaveis = df_metricas.columns[1:-1]  # Ignorar coluna de índice e target
        df_importancia = pd.DataFrame({"Variável": nomes_variaveis, "Importância": importancia})
        df_importancia = df_importancia.sort_values(by="Importância", ascending=False)
        
        fig_importancia = px.bar(
            df_importancia,
            x="Importância",
            y="Variável",
            orientation="h",
            title="Importância das Variáveis no Modelo",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_importancia, use_container_width=True)
    else:
        st.warning("Modelo não disponível para calcular importância das variáveis.")

# -----------------------------------------------------------
# ABA 2 – ANÁLISE EXPLORATÓRIA
# -----------------------------------------------------------
if aba.startswith("📊"):
    st.markdown('<div class="main-header">📊 Análise Exploratória</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Esta seção permite explorar os dados utilizados no treinamento do modelo. 
    Você pode visualizar a distribuição de variáveis e sua relação com o churn.
    """)
    
    # Carregar dados para análise exploratória
    @st.cache_data
    def carregar_dados_exploratorios():
        df = load_data_with_features()
        if df is not None:
            return df.sample(min(500, len(df)))  # Amostra para não sobrecarregar o sistema
        return None
    
    df_exploratorio = carregar_dados_exploratorios()
    
    if df_exploratorio is not None:
        st.subheader("📋 Amostra dos Dados")
        st.write(df_exploratorio)
        
        st.subheader("📊 Distribuição das Variáveis")
        for coluna in df_exploratorio.columns:
            if df_exploratorio[coluna].dtype in ["int64", "float64"]:
                fig = px.histogram(df_exploratorio, x=coluna, nbins=30, title=f"Distribuição de {coluna}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fix for ValueError: Invalid column name 'index'
                fig = px.bar(
                    df_exploratorio[coluna].value_counts().reset_index(),
                    x='index',  # Corrected to use the column created by reset_index()
                    y=coluna,
                    title=f'Distribuição de {coluna}'
                )
                st.plotly_chart(fig, use_container_width=True, key=f'fig_{coluna}')  # Added unique key
        
        st.subheader("🔍 Correlação entre Variáveis")
        fig_corr = px.imshow(df_exploratorio.corr(), title="Mapa de Correlação", color_continuous_scale="RdBu")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("📈 Tendências ao Longo do Tempo")
        if "Ano" in df_exploratorio.columns and "Churn" in df_exploratorio.columns:
            df_tendencias = df_exploratorio.groupby("Ano")["Churn"].mean().reset_index()
            fig_tendencias = px.line(df_tendencias, x="Ano", y="Churn", title="Tendência de Churn ao Longo do Tempo")
            st.plotly_chart(fig_tendencias, use_container_width=True)
        else:
            st.warning("Colunas 'Ano' e/ou 'Churn' não encontradas para análise de tendências.")
    else:
        st.warning("Dados exploratórios não disponíveis.")

# -----------------------------------------------------------
# ABA 3 – EXEMPLOS PRÁTICOS
# -----------------------------------------------------------
if aba.startswith("👥"):
    st.markdown('<div class="main-header">👥 Exemplos Práticos</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Nesta seção, você pode ver exemplos reais de previsões do modelo. 
    Isso ajudará a entender como interpretar as saídas do sistema.
    """)
    
    # Exibir exemplos práticos
    @st.cache_data
    def carregar_exemplos():
        df = load_data_with_features()
        if df is not None:
            return df.sample(min(10, len(df)))  # Amostra para exemplos práticos
        return None
    
    df_exemplos = carregar_exemplos()
    
    if df_exemplos is not None:
        for i, row in df_exemplos.iterrows():
            st.subheader(f"Exemplo {i+1}")
            st.write(row.to_frame().T)
            
            probabilidade, classe = prever_cliente(row)
            st.metric("Probabilidade de Churn", f"{probabilidade:.2f}%", delta_color="inverse")
            st.metric("Classe Prevista", "Churn" if classe == 1 else "Não Churn")
            
            fig_gauge = criar_gauge_chart(probabilidade, "Probabilidade de Churn")
            st.plotly_chart(fig_gauge, use_container_width=True, key=f'fig_gauge_{i}')  # Added unique key
    else:
        st.warning("Exemplos práticos não disponíveis.")

# -----------------------------------------------------------
# ABA 4 – SIMULADOR INDIVIDUAL
# -----------------------------------------------------------
if aba.startswith("👤"):
    st.markdown('<div class="main-header">👤 Simulador Individual</div>', unsafe_allow_html=True)
    
    st.markdown("""
    O simulador abaixo permite testar a probabilidade de churn para um cliente específico. 
    Ajuste as variáveis e veja o impacto na previsão.
    """)
    
    # Formulário para entrada de dados do cliente
    with st.form("form_simulador"):
        st.subheader("📊 Dados do Cliente")
        
        idade = st.slider("Idade do Cliente", 18, 100, 30)
        dependentes = st.slider("Número de Dependentes", 0, 10, 2)
        meses_relacionamento = st.slider("Meses de Relacionamento", 0, 100, 12)
        qtd_produtos = st.slider("Quantidade de Produtos com o Banco", 1, 10, 3)
        meses_inativo = st.slider("Meses Inativo (12 meses)", 0, 12, 0)
        contatos_banco = st.slider("Contatos com o Banco (12 meses)", 0, 10, 2)
        limite_credito = st.slider("Limite de Crédito", 0, 100000, 5000)
        saldo_rotativo = st.slider("Saldo Rotativo", 0, 50000, 1000)
        variacao_valor = st.slider("Variação de Valor Q4/Q1", 0.0, 1.0, 0.1)
        valor_total_transacionado = st.slider("Valor Total Transacionado (12 meses)", 0, 100000, 5000)
        numero_transacoes = st.slider("Número de Transações (12 meses)", 1, 1000, 100)
        variacao_transacoes = st.slider("Variação de Transações Q4/Q1", 0.0, 1.0, 0.1)
        utilizacao_media_limite = st.slider("Utilização Média do Limite", 0.0, 1.0, 0.3)
        
        # Botão para simular
        submitted = st.form_submit_button("🔍 Simular Probabilidade de Churn")
        
        if submitted:
            # Montar row para previsão
            row_simulacao = {
                "Customer_Age": idade,
                "Dependent_count": dependentes,
                "Months_on_book": meses_relacionamento,
                "Total_Relationship_Count": qtd_produtos,
                "Months_Inactive_12_mon": meses_inativo,
                "Contacts_Count_12_mon": contatos_banco,
                "Credit_Limit": limite_credito,
                "Total_Revolving_Bal": saldo_rotativo,
                "Total_Amt_Chng_Q4_Q1": variacao_valor,
                "Total_Trans_Amt": valor_total_transacionado,
                "Total_Trans_Ct": numero_transacoes,
                "Total_Ct_Chng_Q4_Q1": variacao_transacoes,
                "Avg_Utilization_Ratio": utilizacao_media_limite,
            }
            
            # Prever
            probabilidade, classe = prever_cliente(row_simulacao)
            
            # Resultados
            st.markdown("---")
            st.subheader("Resultados da Simulação")
            st.metric("Probabilidade de Churn", f"{probabilidade:.2f}%", delta_color="inverse")
            st.metric("Classe Prevista", "Churn" if classe == 1 else "Não Churn")
            
            fig_gauge = criar_gauge_chart(probabilidade, "Probabilidade de Churn")
            st.plotly_chart(fig_gauge, use_container_width=True, key='fig_gauge_simulacao')
            
            st.markdown("---")
            st.subheader("🔄 Comparar com Outros Cenários")
            st.markdown("""
            Você pode ajustar os parâmetros acima para simular diferentes cenários e ver como 
            isso afeta a probabilidade de churn.
            """)
    
    st.markdown("---")
    
    st.subheader("📚 Exemplos de Clientes")
    st.markdown("""
    Abaixo estão alguns exemplos de clientes com suas respectivas probabilidades de churn. 
    Você pode clicar em um exemplo para carregar os dados no simulador acima.
    """)
    
    # Carregar exemplos de clientes
    @st.cache_data
    def carregar_exemplos_clientes():
        df = load_data_with_features()
        if df is not None:
            return df.sample(min(10, len(df)))  # Amostra para exemplos de clientes
        return None
    
    df_exemplos_clientes = carregar_exemplos_clientes()
    
    if df_exemplos_clientes is not None:
        for i, row in df_exemplos_clientes.iterrows():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image("https://img.icons8.com/fluency/96/user-male-circle.png", width=50)
            
            with col2:
                st.markdown(f"**Cliente {i+1}**")
                st.markdown(f"📅 Idade: {row['Customer_Age']} anos")
                st.markdown(f"👨‍👩‍👦 Dependentes: {row['Dependent_count']}")
                st.markdown(f"📆 Relacionamento: {row['Months_on_book']} meses")
                st.markdown(f"💳 Limite de Crédito: R$ {row['Credit_Limit']:,.2f}")
                st.markdown(f"📉 Saldo Rotativo: R$ {row['Total_Revolving_Bal']:,.2f}")
                st.markdown(f"📊 Variação de Valor Q4/Q1: {row['Total_Amt_Chng_Q4_Q1']:.2f}")
                st.markdown(f"🔄 Transações por Mês: {row['Transacoes_por_Mes']:.2f}")
                st.markdown(f"💰 Gasto Médio Mensal: R$ {row['Gasto_Medio_Mensal']:,.2f}")
                st.markdown(f"⚖️ Uso do Rotativo (Ratio): {row['Rotativo_Ratio']:.2f}")
                st.markdown(f"📉 Disponibilidade Relativa de Limite: {row['Disponibilidade_Relativa']:.2f}")
                
                probabilidade, classe = prever_cliente(row)
                st.metric("Probabilidade de Churn", f"{probabilidade:.2f}%", delta_color="inverse")
                st.metric("Classe Prevista", "Churn" if classe == 1 else "Não Churn")
                
                fig_gauge = criar_gauge_chart(probabilidade, "Probabilidade de Churn")
                st.plotly_chart(fig_gauge, use_container_width=True, key=f'fig_gauge_cliente_{i}')
    
    else:
        st.warning("Exemplos de clientes não disponíveis.")

# -----------------------------------------------------------
# ABA 5 – ANÁLISE EM LOTE
# -----------------------------------------------------------
if aba.startswith("📂"):
    st.markdown('<div class="main-header">📂 Análise em Lote</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Esta seção permite fazer upload de uma lista de clientes e obter previsões em massa. 
    Você receberá um relatório detalhado com insights sobre cada cliente.
    """)
    
    # Formulário para upload de arquivo
    with st.form("form_upload_lote"):
        st.subheader("📤 Upload do Arquivo")
        
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        
        # Botão para processar o arquivo
        submitted = st.form_submit_button("📊 Processar Análise em Lote")
        
        if submitted and uploaded_file is not None:
            # Ler o arquivo CSV
            try:
                df_lote = pd.read_csv(uploaded_file)
                st.success("✅ Arquivo carregado com sucesso!")
                
                # Exibir amostra dos dados
                st.subheader("📋 Amostra dos Dados Carregados")
                st.write(df_lote.head())
                
                # Processar cada cliente
                resultados = []
                for i, row in df_lote.iterrows():
                    probabilidade, classe = prever_cliente(row)
                    resultados.append({
                        "Cliente": i+1,
                        "Probabilidade de Churn": probabilidade,
                        "Classe Prevista": "Churn" if classe == 1 else "Não Churn",
                    })
                
                # Criar DataFrame com resultados
                df_resultados = pd.DataFrame(resultados)
                
                # Exibir resultados
                st.subheader("📊 Resultados da Análise em Lote")
                st.write(df_resultados)
                
                # Download do relatório
                @st.cache_data
                def gerar_relatorio():
                    # Criar um arquivo Excel com os resultados
                    from xlsxwriter import Workbook
                    
                    caminho_arquivo = PROJECT_ROOT / "relatorio_analise_lote.xlsx"
                    workbook = Workbook(caminho_arquivo)
                    worksheet = workbook.add_worksheet("Resultados")
                    
                    # Escrever cabeçalho
                    worksheet.write(0, 0, "Cliente")
                    worksheet.write(0, 1, "Probabilidade de Churn")
                    worksheet.write(0, 2, "Classe Prevista")
                    
                    # Escrever dados
                    for i, resultado in enumerate(resultados):
                        worksheet.write(i+1, 0, resultado["Cliente"])
                        worksheet.write(i+1, 1, resultado["Probabilidade de Churn"])
                        worksheet.write(i+1, 2, resultado["Classe Prevista"])
                    
                    workbook.close()
                    return caminho_arquivo
                
                caminho_relatorio = gerar_relatorio()
                
                st.markdown("---")
                st.subheader("📥 Download do Relatório")
                st.markdown(f"Seu relatório está pronto! [Clique aqui para baixar]({caminho_relatorio})")
                
            except Exception as e:
                st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
    
    st.markdown("---")
    
    st.subheader("📚 Exemplos de Arquivos de Entrada")
    st.markdown("""
    Abaixo estão alguns exemplos de arquivos CSV que podem ser utilizados para a análise em lote. 
    Você pode baixar os exemplos e usar como modelo para seus próprios arquivos.
    """)
    
    # Links para exemplos de arquivos
    exemplos_arquivos = [
        {"nome": "Exemplo 1", "caminho": "https://example.com/exemplo1.csv"},
        {"nome": "Exemplo 2", "caminho": "https://example.com/exemplo2.csv"},
    ]
    
    for exemplo in exemplos_arquivos:
        st.markdown(f"- [{exemplo['nome']}]({exemplo['caminho']})")
    
    st.markdown("---")
    
    st.subheader("📖 Documentação da API")
    st.markdown("""
    Esta seção fornece informações sobre a API utilizada para o modelo de predição de churn. 
    Você encontrará detalhes sobre os endpoints, parâmetros e exemplos de uso.
    """)
    
    # Exibir documentação da API
    try:
        import yaml
        from yaml.loader import SafeLoader
        
        # Carregar documentação da API em YAML
        with open(PROJECT_ROOT / "docs" / "api_documentation.yaml", "r") as file:
            docs_yaml = yaml.load(file, Loader=SafeLoader)
        
        # Exibir documentação formatada
        for endpoint in docs_yaml["endpoints"]:
            st.markdown(f"### {endpoint['path']}")
            st.markdown(f"**Método:** {endpoint['method']}")
            st.markdown(f"**Descrição:** {endpoint['description']}")
            
            st.markdown("**Parâmetros:**")
            for param in endpoint["parameters"]:
                st.markdown(f"- `{param['name']}`: {param['description']} (Tipo: {param['type']})")
            
            st.markdown("**Exemplo de Requisição:**")
            st.code(endpoint["example_request"], language="http")
            
            st.markdown("**Exemplo de Resposta:**")
            st.code(endpoint["example_response"], language="json")
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar documentação da API: {str(e)}")

