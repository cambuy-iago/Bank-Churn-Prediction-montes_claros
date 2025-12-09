# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb

# # Importar do nosso módulo
# from src.features import criar_variaveis_derivadas
# from src.config import DATA_PATH, MODELS_PATH

# # Carregar e preparar os dados
# df = pd.read_csv(DATA_PATH)
# df = criar_variaveis_derivadas(df)

# # Definir features e target
# features = [
#     'Customer_Age', 'Dependent_count', 'Credit_Limit',
#     'Total_Trans_Amt', 'Total_Trans_Ct', 'Ticket_Medio',
#     'Gasto_Medio_Mensal', 'Rotativo_Ratio', 'Score_Relacionamento',
#     'LTV_Proxy', 'Caiu_Valor', 'Caiu_Transacoes'
# ]

# X = df[features]
# y = df["Attrition_Flag"].map({"Attrited Customer": 1, "Existing Customer": 0})

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# # Treinar modelo final
# modelo_final = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
# modelo_final.fit(X_train, y_train)

# # Salvar o modelo
# caminho_modelo = MODELS_PATH / "model_final.pkl"
# joblib.dump(modelo_final, caminho_modelo)
# print(f"[OK] Modelo salvo em: {caminho_modelo}")


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# Importar do nosso módulo
from src.features import criar_variaveis_derivadas
from src.config import DATA_PATH, MODELS_PATH

# Carregar e preparar os dados
df = pd.read_csv(DATA_PATH)
df = criar_variaveis_derivadas(df)

# Definir features (escolha as 12 mais importantes)
features = [
    'Customer_Age', 'Dependent_count', 'Credit_Limit',
    'Total_Trans_Amt', 'Total_Trans_Ct', 'Ticket_Medio',
    'Gasto_Medio_Mensal', 'Rotativo_Ratio', 'Score_Relacionamento',
    'LTV_Proxy', 'Caiu_Valor', 'Caiu_Transacoes'
]

X = df[features]
y = df["Attrition_Flag"].map({"Attrited Customer": 1, "Existing Customer": 0})

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# Treinar modelo final
modelo_final = lgb.LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)

modelo_final.fit(X_train, y_train)

# Salvar o modelo e o scaler
caminho_modelo = MODELS_PATH / "model_final.pkl"
caminho_scaler = MODELS_PATH / "scaler.pkl"

joblib.dump(modelo_final, caminho_modelo)
joblib.dump(scaler, caminho_scaler)

print(f"[OK] Modelo salvo em: {caminho_modelo}")
print(f"[OK] Scaler salvo em: {caminho_scaler}")

# Avaliar o modelo
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

y_pred = modelo_final.predict(X_test)
y_proba = modelo_final.predict_proba(X_test)[:, 1]

print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC ROC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))