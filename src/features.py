# features.py atualizado
import pandas as pd
import numpy as np

def criar_variaveis_derivadas(df):
    """
    Cria variáveis derivadas relevantes para previsão de churn em cartões de crédito.
    Baseado em padrões comportamentais, financeiros e de relacionamento.
    """
    
    # Criar uma cópia para evitar warnings
    df = df.copy()
    
    # ============================================================================
    # 1. VARIÁVEIS DE ATIVIDADE E ENGAJAMENTO
    # ============================================================================
    
    # Ticket médio e intensidade de uso (com tratamento de divisão por zero)
    df['Ticket_Medio'] = np.where(df['Total_Trans_Ct'] != 0, 
                                  df['Total_Trans_Amt'] / df['Total_Trans_Ct'], 
                                  0)
    
    df['Transacoes_por_Mes'] = np.where(df['Months_on_book'] != 0, 
                                        df['Total_Trans_Ct'] / df['Months_on_book'], 
                                        0)
    
    df['Gasto_Medio_Mensal'] = np.where(df['Months_on_book'] != 0, 
                                        df['Total_Trans_Amt'] / df['Months_on_book'], 
                                        0)
    
    # Variação de atividade
    df['Taxa_Queda_Transacoes'] = 1 - df['Total_Ct_Chng_Q4_Q1']
    df['Taxa_Queda_Valor'] = 1 - df['Total_Amt_Chng_Q4_Q1']
    df['Caiu_Transacoes'] = (df['Total_Ct_Chng_Q4_Q1'] < 1).astype(int)
    df['Caiu_Valor'] = (df['Total_Amt_Chng_Q4_Q1'] < 1).astype(int)
    df['Queda_Severa'] = ((df['Total_Ct_Chng_Q4_Q1'] < 0.5) | 
                          (df['Total_Amt_Chng_Q4_Q1'] < 0.5)).astype(int)
    
    # Indicadores de inatividade (usando percentis)
    df['Cliente_Inativo'] = (df['Total_Trans_Ct'] < df['Total_Trans_Ct'].quantile(0.25)).astype(int)
    df['Gasto_Baixo'] = (df['Total_Trans_Amt'] < df['Total_Trans_Amt'].quantile(0.25)).astype(int)
    
    # ============================================================================
    # 2. VARIÁVEIS DE UTILIZAÇÃO DE CRÉDITO E RISCO
    # ============================================================================
    
    # Utilização do crédito
    df['Rotativo_Ratio'] = np.where(df['Credit_Limit'] != 0, 
                                    df['Total_Revolving_Bal'] / df['Credit_Limit'], 
                                    0)
    
    # Criar Avg_Open_To_Buy se não existir
    if 'Avg_Open_To_Buy' not in df.columns:
        df['Avg_Open_To_Buy'] = df['Credit_Limit'] - df['Total_Revolving_Bal']
    
    df['Disponibilidade_Relativa'] = np.where(df['Credit_Limit'] != 0, 
                                              df['Avg_Open_To_Buy'] / df['Credit_Limit'], 
                                              0)
    
    df['Utilizacao_Alta'] = (df['Rotativo_Ratio'] > 0.7).astype(int)
    df['Utilizacao_Baixa'] = (df['Rotativo_Ratio'] < 0.1).astype(int)
    df['Sem_Uso_Rotativo'] = (df['Total_Revolving_Bal'] == 0).astype(int)
    
    # Relação entre gasto e limite
    df['Gasto_vs_Limite'] = np.where(df['Credit_Limit'] != 0, 
                                     df['Total_Trans_Amt'] / df['Credit_Limit'], 
                                     0)
    df['Limite_Subutilizado'] = (df['Gasto_vs_Limite'] < 0.1).astype(int)
    
    # ============================================================================
    # 3. VARIÁVEIS DE RELACIONAMENTO E VALOR DO CLIENTE
    # ============================================================================
    
    # Score de relacionamento
    df['Score_Relacionamento'] = df['Total_Relationship_Count'] + (df['Months_on_book'] / 12)
    
    # Relacionamento fraco
    df['Relacionamento_Fraco'] = ((df['Total_Relationship_Count'] <= 2) & 
                                   (df['Months_on_book'] < 24)).astype(int)
    
    # Cliente multiproduto
    df['Multiproduto'] = (df['Total_Relationship_Count'] >= 4).astype(int)
    
    # Proxy de LTV
    df['LTV_Proxy'] = df['Gasto_Medio_Mensal'] * df['Months_on_book']
    
    # Taxa de produtos por ano
    df['Produtos_por_Ano'] = np.where(df['Months_on_book'] != 0, 
                                      df['Total_Relationship_Count'] / (df['Months_on_book'] / 12), 
                                      0)
    
    # Engajamento recente vs histórico
    df['Transacoes_Recentes_vs_Media'] = np.where(df['Months_on_book'] != 0, 
                                                  df['Total_Trans_Ct'] / (df['Months_on_book'] * 4), 
                                                  0)
    
    # ============================================================================
    # 4. VARIÁVEIS DE SEGMENTAÇÃO E PERFIL
    # ============================================================================
    
    # Faixa etária
    df['Faixa_Idade'] = pd.cut(df['Customer_Age'],
                               bins=[17, 30, 45, 60, 99],
                               labels=['18-30', '31-45', '46-60', '60+'])
    
    df['Cliente_Jovem'] = (df['Customer_Age'] < 35).astype(int)
    df['Cliente_Senior'] = (df['Customer_Age'] >= 60).astype(int)
    
    # Classificação ordinal de renda
    mapa_renda = {
        'Less than $40K': 1,
        '$40K - $60K': 2,
        '$60K - $80K': 3,
        '$80K - $120K': 4,
        '$120K +': 5
    }
    df['Renda_Class'] = df['Income_Category'].map(mapa_renda)
    
    # Segmento de valor
    df['Valor_Cliente'] = df['Renda_Class'] * df['Gasto_Medio_Mensal']
    df['Cliente_Alto_Valor'] = (df['Valor_Cliente'] > df['Valor_Cliente'].quantile(0.75)).astype(int)
    
    # ============================================================================
    # 5. VARIÁVEIS DE PADRÃO E ANOMALIA
    # ============================================================================
    
    # Variação extrema
    df['Variacao_Extrema'] = (
        (np.abs(df['Total_Ct_Chng_Q4_Q1'] - 1) > 0.5) |
        (np.abs(df['Total_Amt_Chng_Q4_Q1'] - 1) > 0.5)
    ).astype(int)
    
    # Inconsistência entre transações e valor
    df['Inconsistencia_Uso'] = np.abs(df['Total_Ct_Chng_Q4_Q1'] - df['Total_Amt_Chng_Q4_Q1'])
    
    # Cliente com limite desproporcional
    df['Limite_Desproporcional'] = (df['Credit_Limit'] > (df['Total_Trans_Amt'] * 10)).astype(int)
    
    # ============================================================================
    # 6. VARIÁVEIS DE CONTATOS E ATENDIMENTO
    # ============================================================================
    
    # Frequência de contatos
    df['Contatos_por_Ano'] = np.where(df['Months_on_book'] != 0, 
                                      df['Contacts_Count_12_mon'] / (df['Months_on_book'] / 12), 
                                      0)
    
    df['Excesso_Contatos'] = (df['Contacts_Count_12_mon'] > 4).astype(int)
    df['Nenhum_Contato'] = (df['Contacts_Count_12_mon'] == 0).astype(int)
    
    # ============================================================================
    # 7. VARIÁVEIS DE DEPENDÊNCIA E INATIVIDADE
    # ============================================================================
    
    # Taxa de inatividade
    df['Taxa_Inatividade'] = df['Months_Inactive_12_mon'] / 12
    df['Altamente_Inativo'] = (df['Months_Inactive_12_mon'] >= 4).astype(int)
    
    # Cliente em risco
    df['Cliente_Risco'] = (
        (df['Caiu_Transacoes'] == 1) &
        (df['Months_Inactive_12_mon'] >= 3) &
        (df['Total_Relationship_Count'] <= 2)
    ).astype(int)
    
    # ============================================================================
    # 8. VARIÁVEIS DE TEMPO E CICLO DE VIDA
    # ============================================================================
    
    # Fase do relacionamento
    df['Fase_Relacionamento'] = pd.cut(df['Months_on_book'],
                                       bins=[0, 12, 24, 36, 999],
                                       labels=['Novo', 'Estabelecido', 'Maduro', 'Veterano'])
    
    df['Cliente_Novo'] = (df['Months_on_book'] <= 12).astype(int)
    
    # Tempo médio entre transações
    df['Dias_Entre_Transacoes'] = np.where(df['Total_Trans_Ct'] != 0, 
                                           (df['Months_on_book'] * 30) / df['Total_Trans_Ct'], 
                                           0)
    
    # ============================================================================
    # 9. SCORES COMPOSTOS
    # ============================================================================
    
    # Score de engajamento (0-1, quanto maior, melhor)
    # NOTA: Em produção, considere usar valores pré-calculados do conjunto de treino
    total_trans_ct_max = df['Total_Trans_Ct'].max() if df['Total_Trans_Ct'].max() != 0 else 1
    total_relationship_count_max = df['Total_Relationship_Count'].max() if df['Total_Relationship_Count'].max() != 0 else 1
    
    df['Score_Engajamento'] = (
        (df['Total_Trans_Ct'] / total_trans_ct_max) * 0.4 +
        (df['Total_Relationship_Count'] / total_relationship_count_max) * 0.3 +
        ((12 - df['Months_Inactive_12_mon']) / 12) * 0.3
    )
    
    # Score de deterioração (0-1, quanto maior, pior)
    df['Score_Deterioracao'] = (
        df['Taxa_Queda_Transacoes'] * 0.4 +
        df['Taxa_Queda_Valor'] * 0.4 +
        df['Taxa_Inatividade'] * 0.2
    )
    
    # Risk Score composto
    df['Risk_Score'] = (
        df['Cliente_Inativo'] * 2 +
        df['Caiu_Transacoes'] * 3 +
        df['Altamente_Inativo'] * 2 +
        df['Relacionamento_Fraco'] * 1 +
        df['Excesso_Contatos'] * 1
    )
    
    return df


def criar_flags_contextuais(df):
    """
    Cria flags adicionais baseadas em contextos específicos de negócio
    """
    
    # Perfis de risco específicos
    df['Perfil_Dormiente'] = (
        (df['Total_Trans_Ct'] < 10) & 
        (df['Months_Inactive_12_mon'] >= 6)
    ).astype(int)
    
    df['Perfil_Transacionador'] = (
        (df['Total_Trans_Ct'] > df['Total_Trans_Ct'].quantile(0.75)) &
        (df['Total_Revolving_Bal'] == 0)
    ).astype(int)
    
    df['Perfil_Rotativo'] = (
        (df['Total_Revolving_Bal'] > 0) &
        (df['Rotativo_Ratio'] > 0.5)
    ).astype(int)
    
    df['Perfil_Premium'] = (
        (df['Credit_Limit'] > df['Credit_Limit'].quantile(0.9)) &
        (df['Total_Trans_Amt'] > df['Total_Trans_Amt'].quantile(0.75))
    ).astype(int)
    
    return df


# Versão simplificada para uso no Streamlit (sem dependências de quantis)
def criar_variaveis_derivadas_simples(df):
    """
    Versão simplificada para uso no Streamlit, sem dependência de quantis
    """
    df = df.copy()
    
    # 1. Features básicas
    df['Ticket_Medio'] = np.where(df['Total_Trans_Ct'] != 0, 
                                  df['Total_Trans_Amt'] / df['Total_Trans_Ct'], 
                                  0)
    
    df['Transacoes_por_Mes'] = np.where(df['Months_on_book'] != 0, 
                                        df['Total_Trans_Ct'] / df['Months_on_book'], 
                                        0)
    
    df['Gasto_Medio_Mensal'] = np.where(df['Months_on_book'] != 0, 
                                        df['Total_Trans_Amt'] / df['Months_on_book'], 
                                        0)
    
    # 2. Utilização de crédito
    df['Rotativo_Ratio'] = np.where(df['Credit_Limit'] != 0, 
                                    df['Total_Revolving_Bal'] / df['Credit_Limit'], 
                                    0)
    
    if 'Avg_Open_To_Buy' not in df.columns:
        df['Avg_Open_To_Buy'] = df['Credit_Limit'] - df['Total_Revolving_Bal']
    
    df['Disponibilidade_Relativa'] = np.where(df['Credit_Limit'] != 0, 
                                              df['Avg_Open_To_Buy'] / df['Credit_Limit'], 
                                              0)
    
    # 3. Flags de variação
    df['Caiu_Transacoes'] = (df['Total_Ct_Chng_Q4_Q1'] < 1).astype(int)
    df['Caiu_Valor'] = (df['Total_Amt_Chng_Q4_Q1'] < 1).astype(int)
    
    # 4. Relacionamento
    df['Score_Relacionamento'] = df['Total_Relationship_Count']
    df['LTV_Proxy'] = df['Gasto_Medio_Mensal'] * df['Months_on_book']
    
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
    
    df['Faixa_Idade'] = df['Customer_Age'].apply(faixa_idade)
    
    # 6. Classificação de renda
    def renda_class(ic):
        if ic in ["$60K - $80K", "$80K - $120K", "$120K +"]:
            return "Alta"
        elif ic in ["$40K - $60K", "$20K - $40K"]:
            return "Média"
        else:
            return "Baixa"
    
    df['Renda_Class'] = df['Income_Category'].apply(renda_class)
    
    return df