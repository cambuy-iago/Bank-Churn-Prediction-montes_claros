"""
Bank Churn Prediction - Pipeline Completo e Organizado
=====================================================

Padrão CRISP-DM:
1. Configuração e imports
2. Carregamento + engenharia de atributos
3. Split treino/teste
4. Pré-processamento (numérico + categórico)
5. Treinamento com validação cruzada e comparação de modelos
6. Treino final, avaliação em teste e salvamento de artefatos
7. Funções de predição para novos clientes

Arquivo para ser executado direto no VSCode ou terminal:
    python src/pipeline_churn.py
"""

# %% 1. IMPORTS E CONFIGURAÇÃO
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier  # pip install xgboost
from lightgbm import LGBMClassifier  # pip install lightgbm


# ---------------------------------------------------------------------
# CONFIGURAÇÃO DO PROJETO
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectConfig:
    random_state: int = 42
    test_size: float = 0.2

    # caminhos principais (relativos à raiz do projeto)
    project_root: Path = Path(__file__).resolve().parent.parent
    data_file: Path = project_root / "data" / "BankChurners.csv"
    model_dir: Path = project_root / "models"
    figures_dir: Path = project_root / "reports" / "figures"
    reports_dir: Path = project_root / "reports"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model_final.pkl"

    @property
    def metrics_csv_path(self) -> Path:
        return self.reports_dir / "metrics_modelos.csv"


CFG = ProjectConfig()

# Garante que as pastas existem
CFG.model_dir.mkdir(parents=True, exist_ok=True)
CFG.figures_dir.mkdir(parents=True, exist_ok=True)
CFG.reports_dir.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    """Pequeno helper para padronizar logs no console."""
    print(f"[LOG] {msg}")


# %% 2. CARREGAMENTO + FEATURE ENGINEERING
def load_and_engineer_data(csv_path: Path) -> pd.DataFrame:
    """
    Carrega a base e aplica feature engineering:
    - Target binário Attrition
    - Variáveis de comportamento (ticket médio, transações/mês etc.)
    - Faixas de idade e renda.
    """
    log(f"Carregando dados de {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Garante apenas duas classes
    df = df[df["Attrition_Flag"].isin(["Attrited Customer", "Existing Customer"])].copy()
    df["Attrition"] = df["Attrition_Flag"].map(
        {"Attrited Customer": 1, "Existing Customer": 0}
    )

    # Remove colunas técnicas de Naive Bayes, se existirem
    cols_drop = [
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon",
        "Naive_Bayes_Classifier_Attrition_Flag_Income_Category_Age",
    ]
    df.drop(columns=[c for c in cols_drop if c in df.columns], inplace=True, errors="ignore")

    # Variáveis derivadas de comportamento
    df["Ticket_Medio"] = df["Total_Trans_Amt"] / df["Total_Trans_Ct"]
    df["Transacoes_por_Mes"] = df["Total_Trans_Ct"] / df["Months_on_book"]
    df["Gasto_Medio_Mensal"] = df["Total_Trans_Amt"] / df["Months_on_book"]
    df["Rotativo_Ratio"] = df["Total_Revolving_Bal"] / df["Credit_Limit"]
    df["Disponibilidade_Relativa"] = (
        df["Credit_Limit"] - df["Total_Revolving_Bal"]
    ) / df["Credit_Limit"]

    # Flags simples de queda de volume/quantidade (comparação sazonal)
    df["Caiu_Transacoes"] = (
        df["Total_Trans_Ct"] < df["Total_Ct_Chng_Q4_Q1"] * df["Total_Trans_Ct"]
    ).astype(int)
    df["Caiu_Valor"] = (
        df["Total_Trans_Amt"] < df["Total_Amt_Chng_Q4_Q1"] * df["Total_Trans_Amt"]
    ).astype(int)

    # Faixas de idade
    def classificar_idade(x: int) -> str:
        if x < 30:
            return "<30"
        elif x < 50:
            return "30-49"
        elif x < 70:
            return "50-69"
        return "70+"

    df["Faixa_Idade"] = df["Customer_Age"].apply(classificar_idade)

    # Faixas de renda agregadas
    def classificar_renda(x: str) -> str:
        if x in ["$60K - $80K", "$80K - $120K", "$120K +"]:
            return "Alta"
        if x in ["$40K - $60K", "$20K - $40K"]:
            return "Média"
        return "Baixa"

    df["Renda_Class"] = df["Income_Category"].apply(classificar_renda)

    log(f"Base após engenharia de atributos: {df.shape}")
    return df


# %% 3. FEATURES, TARGET E PRÉ-PROCESSAMENTO
NUMERIC_FEATURES = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    # engenheiradas:
    "Ticket_Medio",
    "Transacoes_por_Mes",
    "Gasto_Medio_Mensal",
    "Rotativo_Ratio",
    "Disponibilidade_Relativa",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Faixa_Idade",
    "Renda_Class",
]


def get_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa dataframe em X (features) e y (target Attrition)."""
    y = df["Attrition"]
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    return X, y


def build_preprocess_pipeline() -> ColumnTransformer:
    """Cria pré-processador (padronização numérica + one-hot categórica)."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


# %% 4. MODELOS E AVALIAÇÃO COM CROSS-VALIDATION
def get_models() -> Dict[str, object]:
    """Define modelos a serem comparados."""
    rs = CFG.random_state

    models = {
        "LogisticRegression": LogisticRegression(
            random_state=rs,
            max_iter=500,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            random_state=rs,
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            random_state=rs,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            random_state=rs,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        ),
    }
    return models


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    cv_splits: int = 5,
) -> pd.DataFrame:
    """
    Executa validação cruzada estratificada para todos os modelos
    e retorna DataFrame com as principais métricas (média das dobras).
    """
    log("Iniciando avaliação comparativa de modelos (cross-validation)...")
    models = get_models()
    results = []

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=CFG.random_state,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    for name, model in models.items():
        log(f"Avaliando modelo: {name} ...")
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        cv_result = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        results.append(
            {
                "modelo": name,
                "roc_auc_mean": cv_result["test_roc_auc"].mean(),
                "roc_auc_std": cv_result["test_roc_auc"].std(),
                "accuracy_mean": cv_result["test_accuracy"].mean(),
                "recall_mean": cv_result["test_recall"].mean(),
                "precision_mean": cv_result["test_precision"].mean(),
                "f1_mean": cv_result["test_f1"].mean(),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by="roc_auc_mean", ascending=False
    )

    # Salva métricas em CSV para usar no relatório
    results_df.to_csv(CFG.metrics_csv_path, index=False)
    log(f"Métricas de modelos salvas em: {CFG.metrics_csv_path}")

    return results_df


# %% 5. TREINO FINAL, AVALIAÇÃO EM TESTE E SALVAMENTO
def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """Plota e salva matriz de confusão."""
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Não churn", "Churn"],
        yticklabels=["Não churn", "Churn"],
        ylabel="Real",
        xlabel="Predito",
        title=f"Matriz de Confusão - {model_name}",
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    fig.tight_layout()

    fig_path = CFG.figures_dir / f"matriz_confusao_{model_name.lower()}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    log(f"Matriz de confusão salva em: {fig_path}")


def plot_roc_curve(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    """Plota e salva curva ROC do modelo final."""
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"Curva ROC - {model_name}")
    fig_path = CFG.figures_dir / f"roc_curve_{model_name.lower()}.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()
    log(f"Curva ROC salva em: {fig_path}")


def train_and_save_best_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Compara modelos por AUC, escolhe o melhor, re-treina em todo treino,
    avalia em teste e salva o pipeline completo em models/model_final.pkl.
    """
    metrics_df = evaluate_models(X_train, y_train, preprocessor)
    best_model_name = metrics_df.iloc[0]["modelo"]
    log(f"✅ Melhor modelo na validação cruzada: {best_model_name}")
    log(f"\n{metrics_df}\n")

    # Recria o melhor modelo e monta pipeline final
    best_model_cls = get_models()[best_model_name]
    best_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", best_model_cls),
        ]
    )

    log("Treinando modelo final no conjunto de treino...")
    best_pipeline.fit(X_train, y_train)

    # Avaliação em teste
    log("Avaliando modelo final no conjunto de teste...")
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    log("📊 Resultados em teste:")
    log(f"ROC AUC : {roc:.4f}")
    log(f"Accuracy: {acc:.4f}")
    log(f"Recall  : {rec:.4f}")
    log(f"Precision: {prec:.4f}")
    log(f"F1      : {f1:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Matriz de confusão e ROC
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, best_model_name)
    plot_roc_curve(best_pipeline, X_test, y_test, best_model_name)

    # Salvar pipeline completo
    joblib.dump(best_pipeline, CFG.model_path)
    log(f"💾 Modelo final salvo em: {CFG.model_path}")

    return best_pipeline, metrics_df


# %% 6. PREDIÇÃO EM NOVOS CLIENTES
def load_trained_pipeline(model_path: Path | None = None) -> Pipeline:
    """Carrega o pipeline treinado (model_final.pkl por padrão)."""
    if model_path is None:
        model_path = CFG.model_path
    return joblib.load(model_path)


def predict_new_customers(
    new_data: pd.DataFrame,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """
    Retorna probabilidade de churn para clientes em new_data.

    new_data deve ter as mesmas colunas NUMERIC_FEATURES + CATEGORICAL_FEATURES
    e já conter as variáveis engenheiradas. Se você tiver a base original,
    reaplique load_and_engineer_data e depois filtre as colunas.
    """
    pipeline_trained = load_trained_pipeline(model_path)
    proba_churn = pipeline_trained.predict_proba(new_data)[:, 1]

    result = new_data.copy()
    result["prob_churn"] = proba_churn
    return result


# %% 7. MAIN
def main() -> None:
    log("🔹 Iniciando pipeline de churn bancário...")

    df = load_and_engineer_data(CFG.data_file)
    X, y = get_feature_target(df)
    log(f"Tamanho final de X: {X.shape}, y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CFG.test_size,
        random_state=CFG.random_state,
        stratify=y,
    )

    preprocessor = build_preprocess_pipeline()

    best_pipeline, metrics_df = train_and_save_best_model(
        X_train, X_test, y_train, y_test, preprocessor
    )

    log("✅ Pipeline concluído com sucesso.")


if __name__ == "__main__":
    main()

# %%
