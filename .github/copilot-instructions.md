# AI Agent Instructions - Bank Churn Prediction

## Project Overview

This is a **Bank Churn Prediction** ML project (MBA capstone) that predicts customer credit card attrition using ensemble models (LightGBM, XGBoost, Random Forest). Architecture follows **CRISP-DM workflow** split between exploratory notebooks and modular Python pipelines.

## Architecture & Data Flow

### Core Components

1. **Data Pipeline** (`src/config.py` + `src/pipeline_churn.py`)
   - Raw data: `data/BankChurners.csv`
   - Config uses **relative paths via Path** for cross-platform compatibility
   - Auto-creates output directories (`reports/figures`, `reports/text`, `models`)

2. **Feature Engineering** (`src/features.py`)
   - Central module: `criar_variaveis_derivadas()` creates 50+ derived features
   - Categories: activity metrics, credit utilization, relationship indicators, decline signals
   - Handles division-by-zero with `np.where()` pattern consistently
   - Features feed into all downstream training scripts

3. **Model Training** (Three parallel scripts)
   - `train_lgbm.py` → LightGBM (AUC: 0.9826) ✅ **Production model**
   - `train_xgb.py` → XGBoost (AUC: 0.9824)
   - `train_rf.py` → Random Forest (AUC: 0.9770)
   - All save models to `models/` as `.pkl` files
   - Metrics saved to `reports/metrics_modelos.csv`

4. **Prediction Interface** (`webapp/app.py`)
   - Streamlit app loads final model from `models/model_final.pkl`
   - Input features match 12-feature subset from training
   - Returns probability + risk classification

### Data Flow Sequence

```
BankChurners.csv 
  ↓ (features.py)
Engineered features (50+ columns)
  ↓ (train_*.py scripts)
Trained models (.pkl)
  ↓ (webapp/app.py)
Streamlit predictions
```

## Key Conventions & Patterns

### Feature Engineering Pattern
```python
# Division by zero handled with np.where() throughout features.py
df['Ticket_Medio'] = np.where(df['Total_Trans_Ct'] != 0, 
                              df['Total_Trans_Amt'] / df['Total_Trans_Ct'], 
                              0)
```
Always use this pattern when creating ratios/derived metrics.

### Model Configuration Pattern
Both `train_model.py` and individual training scripts use **dataclass-based config**:
```python
@dataclass(frozen=True)
class ProjectConfig:
    random_state: int = 42
    test_size: float = 0.2
    project_root: Path = Path(__file__).resolve().parent.parent
```
All paths are relative to project root for portability.

### Target Variable Mapping
Churn target is always binary: `"Attrited Customer" → 1`, `"Existing Customer" → 0`

### Class Imbalance Strategy
**`class_weight='balanced'` is the final chosen strategy** (tested against SMOTE). Use this parameter consistently:
```python
# In LightGBM
lgb.LGBMClassifier(..., is_unbalanced=True, class_weight='balanced')

# In XGBoost
xgb.XGBClassifier(..., scale_pos_weight=weight_ratio)

# In RandomForest
RandomForestClassifier(..., class_weight='balanced')
```
This approach outperformed SMOTE in cross-validation and avoids synthetic data artifacts.

### Model Selection & Feature Subset
Feature set for predictions is **standardized 12-feature baseline**:
`Customer_Age`, `Dependent_count`, `Credit_Limit`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Ticket_Medio`, `Gasto_Medio_Mensal`, `Rotativo_Ratio`, `Score_Relacionamento`, `LTV_Proxy`, `Caiu_Valor`, `Caiu_Transacoes`

**Note**: This is the proven baseline for production. Alternative feature sets can be tested in exploration notebooks, but must revert to this 12-feature subset for model comparisons and deployment.

### Model Versioning & Evaluation Logging
Use `src/model_versioning.py` (ModelVersionManager class) for all model workflows:
- **Naming convention**: `model_{algorithm}_{version}.pkl` (e.g., `model_lgbm_v1.pkl`, `model_xgb_v2.pkl`)
- **Automatic logging**: `models/versions_log.csv` tracks filename, algorithm, AUC, accuracy, precision, recall, F1, timestamp, notes
- **Production model**: `model_final.pkl` auto-updated when `is_production=True` during save
- **Usage in scripts**: Import `ModelVersionManager`, `ModelMetrics` from `model_versioning.py`; see `train_lgbm_enhanced.py` for template

```python
from src.model_versioning import ModelVersionManager, ModelMetrics

manager = ModelVersionManager(models_dir)
metrics = ModelMetrics(algorithm='lgbm', version=None, auc=0.9826, ...)
manager.save_model(model, algorithm='lgbm', metrics=metrics, is_production=True)
```

## Notebook Workflow (Recommended Execution Order)

Follow this sequence for reproducible analysis and model development:

1. **0_Import_Tratamento.ipynb** → Data loading, cleaning, initial checks
2. **EDA_Cluster_PCA.ipynb** → Exploratory analysis, dimensionality reduction insights
3. **Model_Training.ipynb** → Baseline model training (XGB, LGBM, RF)
4. **Feature_Importance_SHAP.ipynb** → Interpretability analysis (SHAP values)
5. **Balancing_And_Tuning.ipynb** → Hyperparameter tuning, class weight optimization
6. **Final_Model_Selection.ipynb** → Champion model selection and validation

**Note**: Multiple EDA variants exist (`1_Analise_Exploratoria.ipynb`, `Análise_Exploratória_Dados(EDA).ipynb`). Use the primary sequence above for consistency. Model Analysis notebooks (`LightGBM_Model_Analysis.ipynb`, etc.) are reference implementations.

## Critical Developer Workflows

### Running the Full Pipeline
```powershell
# Activate venv first
.\.venv\Scripts\Activate.ps1

# Run complete pipeline (CRISP-DM end-to-end)
python src/pipeline_churn.py

# Or train individual models with feature engineering
python src/train_lgbm.py
python src/train_xgb.py
python src/train_rf.py
```

### Launching Prediction UI
```powershell
streamlit run webapp/app.py
```
The Streamlit app (`webapp/app.py`) provides:
- **Organized input sections** by profile type (demographic, activity, risk indicators)
- **Real-time predictions** with risk classification (low/medium/high)
- **Feature importance visualization** showing top 5 most influential features
- **Business recommendations** based on client profile (e.g., engagement strategies)
- **Export functionality** (JSON/CSV) for audit trail and CRM integration
- **Model metrics dashboard** with AUC, accuracy, and model info
- **Colorized risk gauge** for intuitive decision-making

### Dependencies
- ML: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`
- Viz: `matplotlib`, `seaborn`
- Web: `streamlit`
See `requirements.txt` for versions.

## Integration Points

1. **Feature → Model**: All training scripts import from `src/features.py`
2. **Config → Paths**: All scripts use `src/config.py` for centralized path management
3. **Notebooks → Production**: Exploratory work→ encoded in `src/*.py` modules
4. **Model → App**: Streamlit loads pre-trained `.pkl` from `models/` directory

## When Adding Features

1. Add new column creation in `src/features.py` (use `np.where()` for safety)
2. Test in notebook first (e.g., `1_Analise_Exploratoria.ipynb`)
3. Add feature name to 12-feature subset in `src/train_lgbm.py`, `webapp/app.py`
4. Re-train and update `models/model_final.pkl`
5. Update webapp input sliders/number_inputs accordingly

## Common Tasks

| Task | Files | Command |
|------|-------|---------|
| Add feature | `src/features.py` | `python src/pipeline_churn.py` |
| Tune hyperparameters | `src/train_lgbm.py` | `python src/train_lgbm.py` |
| New model type | `src/train_*.py` (new file) | Create new training script |
| Deploy predictions | `webapp/app.py` | `streamlit run webapp/app.py` |
| Check metrics | `reports/metrics_modelos.csv` | View after training |
