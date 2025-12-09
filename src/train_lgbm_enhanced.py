"""
Enhanced Model Training Template with Versioning and Logging
============================================================

This template demonstrates best practices for training and evaluating models
with automatic versioning, metrics logging, and organized artifact storage.

Usage:
    python src/train_lgbm_enhanced.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import lightgbm as lgb

# Project imports
from src.features import criar_variaveis_derivadas
from src.config import DATA_PATH, MODELS_PATH, TEXT_PATH, FIGURES_PATH
from src.model_versioning import ModelVersionManager, ModelMetrics, log_evaluation


# ============================================================================
# CONFIGURATION
# ============================================================================
class TrainingConfig:
    """Training configuration."""
    algorithm = "lgbm"
    random_state = 42
    test_size = 0.2
    cv_folds = 5
    
    # Features for prediction (12-feature baseline)
    features = [
        'Customer_Age', 'Dependent_count', 'Credit_Limit',
        'Total_Trans_Amt', 'Total_Trans_Ct', 'Ticket_Medio',
        'Gasto_Medio_Mensal', 'Rotativo_Ratio', 'Score_Relacionamento',
        'LTV_Proxy', 'Caiu_Valor', 'Caiu_Transacoes'
    ]
    
    # LightGBM hyperparameters
    lgbm_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'is_unbalanced': True,
        'class_weight': 'balanced',
        'verbose': -1
    }


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main():
    """Execute complete training pipeline."""
    
    print("\n" + "="*70)
    print("MODEL TRAINING PIPELINE - LightGBM with Versioning")
    print("="*70 + "\n")
    
    # 1. Load and prepare data
    print("[1/5] Carregando dados...")
    df = pd.read_csv(DATA_PATH)
    print(f"     ✓ {len(df)} registros carregados")
    
    # 2. Feature engineering
    print("[2/5] Engenharia de features...")
    df = criar_variaveis_derivadas(df)
    print(f"     ✓ {len(df.columns)} variáveis disponíveis")
    
    # Select features
    X = df[TrainingConfig.features]
    y = df["Attrition_Flag"].map({"Attrited Customer": 1, "Existing Customer": 0})
    print(f"     ✓ Usando 12-feature baseline")
    print(f"     ✓ Distribuição: {(y.sum() / len(y)):.1%} churn, {(1 - y.sum() / len(y)):.1%} retenção")
    
    # 3. Split data
    print("[3/5] Dividindo dados (treino/teste)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TrainingConfig.test_size,
        random_state=TrainingConfig.random_state,
        stratify=y
    )
    print(f"     ✓ Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # 4. Train model
    print("[4/5] Treinando modelo...")
    model = lgb.LGBMClassifier(**TrainingConfig.lgbm_params)
    model.fit(X_train, y_train)
    print(f"     ✓ Modelo treinado com sucesso")
    
    # 5. Evaluate
    print("[5/5] Avaliando modelo...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics_dict = {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print(f"     ✓ AUC-ROC: {metrics_dict['auc']:.4f}")
    print(f"     ✓ Acurácia: {metrics_dict['accuracy']:.4f}")
    print(f"     ✓ Precisão: {metrics_dict['precision']:.4f}")
    print(f"     ✓ Recall: {metrics_dict['recall']:.4f}")
    print(f"     ✓ F1-Score: {metrics_dict['f1']:.4f}")
    
    # ============================================================================
    # VERSIONING & LOGGING
    # ============================================================================
    
    # Initialize version manager
    manager = ModelVersionManager(MODELS_PATH)
    
    # Create metrics object
    model_metrics = ModelMetrics(
        algorithm=TrainingConfig.algorithm,
        version=None,  # Will be auto-assigned
        auc=metrics_dict['auc'],
        accuracy=metrics_dict['accuracy'],
        precision=metrics_dict['precision'],
        recall=metrics_dict['recall'],
        f1=metrics_dict['f1'],
        notes="12-feature baseline with class_weight='balanced'"
    )
    
    # Save model with versioning
    model_path = manager.save_model(
        model=model,
        algorithm=TrainingConfig.algorithm,
        metrics=model_metrics,
        is_production=True  # Also save as model_final.pkl
    )
    
    print(f"\n✅ Modelo salvo com versão: {model_metrics.version}")
    print(f"   Caminho: {model_path}")
    
    # ============================================================================
    # DETAILED EVALUATION & REPORTING
    # ============================================================================
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, 
                                        target_names=['Existing', 'Attrited'],
                                        digits=4)
    
    # Log evaluation
    eval_file = log_evaluation(
        output_dir=TEXT_PATH,
        algorithm=f"lgbm_{model_metrics.version}",
        metrics=metrics_dict,
        report_text=class_report
    )
    
    # ============================================================================
    # VISUALIZATIONS
    # ============================================================================
    
    # 1. Feature Importance
    feature_importance_df = pd.DataFrame({
        'feature': TrainingConfig.features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feature_importance_df)))
    ax.barh(feature_importance_df['feature'], feature_importance_df['importance'], color=colors)
    ax.set_xlabel('Importância', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance - LightGBM', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    
    importance_path = FIGURES_PATH / f"feature_importance_lgbm_{model_metrics.version}.png"
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Feature importance: {importance_path}")
    plt.close()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Existing', 'Attrited'],
                yticklabels=['Existing', 'Attrited'])
    ax.set_xlabel('Predito', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real', fontsize=12, fontweight='bold')
    ax.set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    cm_path = FIGURES_PATH / f"confusion_matrix_lgbm_{model_metrics.version}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix: {cm_path}")
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - LightGBM', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = FIGURES_PATH / f"roc_curve_lgbm_{model_metrics.version}.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"📊 ROC curve: {roc_path}")
    plt.close()
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"✓ Model version: {model_metrics.version}")
    print(f"✓ Algorithm: {TrainingConfig.algorithm.upper()}")
    print(f"✓ AUC-ROC: {metrics_dict['auc']:.4f}")
    print(f"✓ Model path: {model_path}")
    print(f"✓ Evaluation report: {eval_file}")
    print(f"✓ Production model: {MODELS_PATH / 'model_final.pkl'}")
    print("\n📍 Next steps:")
    print("   1. Review metrics and evaluation reports")
    print("   2. Test in webapp: streamlit run webapp/app.py")
    print("   3. Compare with other versions in: models/versions_log.csv")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
