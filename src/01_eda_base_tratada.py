# %% [markdown]
# # 📊 EDA Completa – Base Tratada de Churn Bancário
#
# Notebook-script para VSCode (Python + Jupyter) usando a base:
# `data/base_tratada.csv`
#
# Objetivos:
# - Carregar a base tratada
# - Verificar qualidade dos dados
# - Explorar distribuições (numéricas e categóricas)
# - Analisar outliers
# - Ver correlações com o target
# - Preparar estruturas para PCA e clusterização
#
# Execute célula a célula no VSCode para uma análise didática.

# %% [markdown]
# ## 0. Imports e Configurações Iniciais

# %%
from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy import stats

# Tentativa de importar statsmodels (opcional)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    variance_inflation_factor = None
    add_constant = None

# Tentativa de importar plotly (opcional)
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1200)

print("✅ Imports concluídos.")

# %% [markdown]
# ## 1. Funções Utilitárias de Data Quality e EDA

# %%
def data_quality_report(df: pd.DataFrame, save_path: Path | str | None = None) -> None:
    """
    Imprime verificações rápidas de qualidade de dados e, opcionalmente,
    salva a base tratada em CSV.

    Uso:
        data_quality_report(df, save_path="data/base_tratada.csv")
    """
    if df is None:
        raise ValueError("DataFrame 'df' não foi fornecido (None).")

    print("\n🔍 Verificação de Data Quality:")
    print(f"- Registros: {len(df)}")
    print(f"- Duplicados: {df.duplicated().sum()} registros")

    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"- Memória usada: {mem_mb:.2f} MB")

    # Colunas constantes
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if constant_cols:
        print(f"- Colunas constantes: {constant_cols}")
    else:
        print("- Colunas constantes: nenhuma")

    # Missing values
    total_missing = df.isnull().sum().sum()
    print(f"- Valores ausentes (total): {total_missing}")
    if total_missing:
        missing_per_col = df.isnull().sum()
        top_missing = (
            missing_per_col[missing_per_col > 0]
            .sort_values(ascending=False)
            .head(10)
        )
        print("- Top colunas com NA:")
        for col, cnt in top_missing.items():
            pct = cnt / len(df) * 100
            print(f"    - {col}: {cnt} ({pct:.2f}%)")

    # Zeros em colunas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    zero_counts = {c: int((df[c] == 0).sum()) for c in num_cols if (df[c] == 0).any()}
    if zero_counts:
        print("- Contagens de zeros em colunas numéricas (até 10):")
        for i, (col, cnt) in enumerate(
            sorted(zero_counts.items(), key=lambda x: -x[1])
        ):
            if i >= 10:
                break
            print(f"    - {col}: {cnt}")

    # Salvar arquivo tratado opcionalmente
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"- Base salva em: {p.resolve()}")


def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> dict:
    """Detecta outliers usando o método IQR (quartis)."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]
    outlier_pct = len(outliers) / len(df) * 100

    return {
        "outlier_count": int(len(outliers)),
        "outlier_percentage": float(outlier_pct),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "Q1": float(Q1),
        "Q3": float(Q3),
        "IQR": float(IQR),
        "min": float(df[column].min()),
        "max": float(df[column].max()),
        "mean": float(df[column].mean()),
        "median": float(df[column].median()),
    }


def detailed_statistical_comparison(
    df: pd.DataFrame, numeric_cols: list[str], target_col: str = "Attrition"
) -> pd.DataFrame:
    """Compara estatísticas das variáveis numéricas entre as classes do target."""
    stats_comparison = pd.DataFrame()

    for col in numeric_cols:
        # Estatísticas por grupo
        group_stats = df.groupby(target_col)[col].agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max",
                "skew",
                lambda x: x.quantile(0.75) - x.quantile(0.25),
            ]
        ).rename(columns={"<lambda_0>": "IQR"})

        group0 = df[df[target_col] == 0][col].dropna()
        group1 = df[df[target_col] == 1][col].dropna()

        if len(group0) > 1 and len(group1) > 1:
            t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)
        else:
            t_stat, p_value = np.nan, np.nan

        cv0 = group0.std() / group0.mean() if group0.mean() != 0 else np.nan
        cv1 = group1.std() / group1.mean() if group1.mean() != 0 else np.nan

        stats_comparison[col] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean_diff": group_stats.loc[1, "mean"] - group_stats.loc[0, "mean"],
            "cv_0": cv0,
            "cv_1": cv1,
        }

    return stats_comparison.T


def significant_correlations(
    df: pd.DataFrame, cols: list[str], target_col: str = "Attrition", alpha: float = 0.05
) -> pd.DataFrame:
    """Calcula correlação de Pearson e p-valor entre features numéricas e o target."""
    results = []
    for col in cols:
        if col == target_col:
            continue

        x = df[col].dropna()
        y = df[target_col].loc[x.index].dropna()

        if len(x) > 1 and len(y) > 1:
            corr, p_value = stats.pearsonr(x, y)
        else:
            corr, p_value = np.nan, np.nan

        results.append(
            {
                "feature": col,
                "correlation": corr,
                "p_value": p_value,
                "significant": bool(p_value < alpha) if not np.isnan(p_value) else False,
                "abs_correlation": abs(corr) if not np.isnan(corr) else np.nan,
            }
        )

    return pd.DataFrame(results).sort_values("abs_correlation", ascending=False)


def calculate_vif(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Calcula VIF (Variance Inflation Factor) para avaliar multicolinearidade."""
    if not HAS_STATSMODELS:
        raise ImportError(
            "statsmodels não está instalado. Instale com: pip install statsmodels"
        )

    from statsmodels.tools.tools import add_constant
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = add_constant(df[numeric_cols].dropna())
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)


def pca_3d_visualization(
    X_scaled: np.ndarray, target: pd.Series, n_components: int = 3
) -> tuple[np.ndarray, PCA]:
    """PCA em 3D para visualização dos clientes coloridos por churn."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=target,
        cmap="viridis",
        alpha=0.6,
        s=15,
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    plt.title("PCA 3D - Clientes por Status de Churn")
    plt.legend(*scatter.legend_elements(), title="Churn")
    plt.tight_layout()
    plt.show()

    return X_pca, pca


def elbow_method(X_scaled: np.ndarray, max_clusters: int = 10) -> None:
    """Método do cotovelo para escolher k no KMeans."""
    inertias = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), inertias, marker="o")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inércia (Within-Cluster SSE)")
    plt.title("Método do Cotovelo para KMeans")
    plt.grid(True)
    plt.show()


def silhouette_analysis(X_scaled: np.ndarray, max_clusters: int = 10) -> None:
    """Silhouette score por número de clusters."""
    scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), scores, marker="o", color="tab:red")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Análise de Silhouette para KMeans")
    plt.grid(True)
    plt.show()


def generate_eda_report(df: pd.DataFrame, target_col: str = "Attrition") -> dict:
    """Gera um dicionário com resumo de EDA para exportar em JSON."""
    report = {
        "dataset_shape": df.shape,
        "target_distribution": df[target_col].value_counts().to_dict()
        if target_col in df.columns
        else None,
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_features": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_features": df.select_dtypes(include=["object"]).columns.tolist(),
    }

    report["numeric_stats"] = df.describe().T.to_dict()

    if target_col in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = (
            df[numeric_cols].corr()[target_col].sort_values(ascending=False)
        )
        report["top_correlations"] = correlations.head(10).to_dict()

    return report


def export_analysis_results(
    df: pd.DataFrame,
    cluster_labels: np.ndarray | None,
    pca_result: np.ndarray | None,
    output_dir: str = "eda_results",
) -> None:
    """Exporta resultados principais de EDA para CSVs em uma pasta."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Dataset com clusters e PCA, se fornecidos
    df_export = df.copy()
    if cluster_labels is not None:
        df_export["Cluster"] = cluster_labels
    if pca_result is not None and pca_result.shape[1] >= 2:
        df_export["PC1"] = pca_result[:, 0]
        df_export["PC2"] = pca_result[:, 1]

    df_export.to_csv(f"{output_dir}/dataset_with_clusters.csv", index=False)

    # Resumo estatístico
    df.describe().to_csv(f"{output_dir}/statistical_summary.csv")

    print(f"✅ Resultados exportados para '{output_dir}/'")    


def interactive_distribution(df: pd.DataFrame, col: str, target_col: str = "Attrition") -> None:
    """Histograma interativo por classe (se Plotly estiver instalado)."""
    if not HAS_PLOTLY:
        print("⚠️ Plotly não está instalado. Use: pip install plotly")
        return

    fig = px.histogram(
        df,
        x=col,
        color=target_col,
        marginal="box",
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        title=f"Distribuição de {col} por Churn",
    )
    fig.show()


def visualize_outliers(df: pd.DataFrame, column: str, target_col: str = "Attrition", outlier_stats: dict | None = None) -> None:
    """Boxplots geral e por classe, com impressão das estatísticas de outliers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=df, y=column, ax=axes[0])
    axes[0].set_title(f"Boxplot de {column}")

    if target_col in df.columns:
        sns.boxplot(data=df, x=target_col, y=column, ax=axes[1])
        axes[1].set_title(f"{column} por Status (Attrition)")
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    if outlier_stats is not None and column in outlier_stats:
        stats_col = outlier_stats[column]
        print(f"\n📈 Estatísticas de {column}:")
        for k, v in stats_col.items():
            print(f"  {k}: {v}")

# %% [markdown]
# ## 2. Carregamento da Base `data/base_tratada.csv`

# %%
# Detectar raiz do projeto a partir da pasta atual
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name.lower() in {"notebooks", "nb"}:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = PROJECT_ROOT / "data" / "base_tratada.csv"

print("📁 Projeto em:", PROJECT_ROOT)
print("📄 Procurando base em:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Arquivo base_tratada.csv não encontrado em {DATA_PATH}.\n"
        "Verifique se está rodando o notebook a partir da pasta raiz do projeto "
        "(Bank-Churn-Prediction-montes_claros) ou ajuste o caminho manualmente."
    )

df = pd.read_csv(DATA_PATH)
print("✅ Base carregada! Formato:", df.shape)
try:
    display(df.head())
except NameError:
    print(df.head())

# %% [markdown]
# ## 3. Data Quality Rápido

# %%
data_quality_report(df)

# %% [markdown]
# ## 4. Identificação de Variáveis Numéricas e Categóricas

# %%
TARGET_COL = "Attrition"

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COL in numeric_cols:
    numeric_cols.remove(TARGET_COL)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("🔢 Variáveis numéricas:", numeric_cols)
print("🔠 Variáveis categóricas:", cat_cols)

# %% [markdown]
# ## 5. Distribuição da Variável Target (Churn)

# %%
if TARGET_COL in df.columns:
    churn_counts = df[TARGET_COL].value_counts().sort_index()
    churn_percent = churn_counts / churn_counts.sum() * 100

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=ax[0])
    ax[0].set_title("Distribuição Absoluta de Classes")
    ax[0].set_xlabel("Attrition (0 = ativo, 1 = churn)")
    ax[0].set_ylabel("Quantidade")

    sns.barplot(x=churn_percent.index, y=churn_percent.values, ax=ax[1])
    ax[1].set_title("Distribuição Percentual de Classes")
    ax[1].set_xlabel("Attrition")
    ax[1].set_ylabel("% de clientes")

    for i, p in enumerate(ax[1].patches):
        ax[1].annotate(
            f"{churn_percent.values[i]:.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()

    print("\n📌 Taxa de churn aproximada:", f"{churn_percent.loc[1]:.2f}%" if 1 in churn_percent.index else churn_percent)

# %% [markdown]
# ## 6. Análise de Outliers (IQR)

# %%
outlier_summary: dict[str, dict] = {}

for col in numeric_cols:
    outlier_summary[col] = detect_outliers_iqr(df, col)

outlier_df = pd.DataFrame(outlier_summary).T
outlier_df = outlier_df.sort_values("outlier_percentage", ascending=False)

print("Resumo de outliers (top 10):")
try:
    display(outlier_df.head(10))
except NameError:
    print(outlier_df.head(10))

significant_outliers = outlier_df[outlier_df["outlier_percentage"] > 1]
print("\nVariáveis com mais de 1% de outliers:", len(significant_outliers))
try:
    display(significant_outliers.head(15))
except NameError:
    print(significant_outliers.head(15))

# %% [markdown]
# ### Visualização dos principais outliers

# %%
for col in significant_outliers.head(3).index:
    visualize_outliers(df, col, target_col=TARGET_COL, outlier_stats=outlier_summary)

# %% [markdown]
# ### Outliers por classe (churn vs não churn)

# %%
if TARGET_COL in df.columns:
    outliers_by_class = {}

    for col in significant_outliers.head(5).index:
        churn_0 = df[df[TARGET_COL] == 0][col]
        churn_1 = df[df[TARGET_COL] == 1][col]

        stats_0 = detect_outliers_iqr(pd.DataFrame({col: churn_0}), col)
        stats_1 = detect_outliers_iqr(pd.DataFrame({col: churn_1}), col)

        outliers_by_class[col] = {
            "classe_0_pct": stats_0["outlier_percentage"],
            "classe_1_pct": stats_1["outlier_percentage"],
            "diferença": stats_1["outlier_percentage"] - stats_0["outlier_percentage"],
        }

    outliers_class_df = pd.DataFrame(outliers_by_class).T.sort_values(
        "diferença", ascending=False
    )

    print("Diferença na porcentagem de outliers entre classes:")
    try:
        display(outliers_class_df)
    except NameError:
        print(outliers_class_df)

# %% [markdown]
# ## 7. Correlações com o Target e VIF (opcional)

# %%
corr_results = significant_correlations(df, numeric_cols, target_col=TARGET_COL)
print("Top correlações (em valor absoluto) com Attrition:")
try:
    display(corr_results.head(15))
except NameError:
    print(corr_results.head(15))

if HAS_STATSMODELS:
    try:
        vif_df = calculate_vif(df, numeric_cols)
        print("\nVIF (multicolinearidade) – top 15:")
        try:
            display(vif_df.head(15))
        except NameError:
            print(vif_df.head(15))
    except Exception as e:
        print("⚠️ Erro ao calcular VIF:", e)
else:
    print("⚠️ statsmodels não está instalado. Pulei o cálculo de VIF.")

# %% [markdown]
# ## 8. PCA 2D + Clusterização (estrutura básica)

# %%
# Escalonar apenas variáveis numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

# PCA 2D para visualização
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca_2d, columns=["PC1", "PC2"])
if TARGET_COL in df.columns:
    df_pca["Attrition"] = df[TARGET_COL]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_pca.sample(n=min(3000, len(df_pca)), random_state=42),
    x="PC1",
    y="PC2",
    hue="Attrition" if "Attrition" in df_pca.columns else None,
    alpha=0.6,
)
plt.title("PCA 2D – Clientes por Status de Churn")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### PCA 3D (opcional)

# %%
if len(numeric_cols) >= 3:
    X_pca_3d, pca_model_3d = pca_3d_visualization(
        X_scaled,
        df[TARGET_COL] if TARGET_COL in df.columns else pd.Series([0] * len(df))
    )
else:
    X_pca_3d, pca_model_3d = None, None
    print("PCA 3D não executado: menos de 3 variáveis numéricas.")

# %% [markdown]
# ### Escolha de k com cotovelo e silhouette

# %%
elbow_method(X_scaled, max_clusters=8)
silhouette_analysis(X_scaled, max_clusters=8)

# %% [markdown]
# ### Clusterização com KMeans (k=3 como exemplo)

# %%
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df_clusters = df.copy()
df_clusters["Cluster"] = cluster_labels

cols_for_profile = numeric_cols.copy()
if TARGET_COL in df.columns:
    cols_for_profile.append(TARGET_COL)

cluster_profile = (
    df_clusters.groupby("Cluster")[cols_for_profile]
    .mean()
    .round(3)
)

print("Perfil médio dos clusters (variáveis numéricas + churn):")
try:
    display(cluster_profile)
except NameError:
    print(cluster_profile)

if TARGET_COL in df.columns:
    cluster_churn = (
        df_clusters.groupby("Cluster")[TARGET_COL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "taxa_churn", "count": "qtd_clientes"})
        .round(3)
    )
    print("\nTaxa de churn por cluster:")
    try:
        display(cluster_churn)
    except NameError:
        print(cluster_churn)

# %% [markdown]
# ## 9. Exportar Relatórios e Resultados

# %%
# Gerar JSON de EDA
eda_report = generate_eda_report(df, target_col=TARGET_COL)
with open("eda_report.json", "w", encoding="utf-8") as f:
    json.dump(eda_report, f, indent=4, ensure_ascii=False)
print("📁 Arquivo 'eda_report.json' salvo.")

# Exportar dataset com clusters + resumo estatístico
export_analysis_results(
    df,
    cluster_labels=cluster_labels,
    pca_result=X_pca_2d,
    output_dir="eda_results",
)

print("\n✅ EDA concluída. Você pode agora usar estes insights na etapa de modelagem.")
