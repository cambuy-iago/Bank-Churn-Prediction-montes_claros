"""
Model Versioning and Evaluation Logging System
==============================================

Manages model versioning, metrics tracking, and artifact organization.
Used by all training scripts to maintain reproducible model lifecycle.
"""

import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    algorithm: str
    version: str
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_curve_path: Optional[str] = None
    confusion_matrix_path: Optional[str] = None
    notes: str = ""
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ModelVersionManager:
    """
    Manages model versioning, storage, and metrics logging.
    
    Naming convention: model_{algorithm}_{version}.pkl
    Metrics tracking: models/versions_log.csv
    """
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize version manager.
        
        Args:
            models_dir: Directory to store models (default: project/models)
        """
        if models_dir is None:
            self.models_dir = Path(__file__).resolve().parent.parent / "models"
        else:
            self.models_dir = models_dir
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions_log_path = self.models_dir / "versions_log.csv"
        
        # Ensure versions log exists
        if not self.versions_log_path.exists():
            pd.DataFrame(columns=[
                'filename', 'algorithm', 'version', 'auc', 'accuracy', 
                'precision', 'recall', 'f1', 'timestamp', 'notes'
            ]).to_csv(self.versions_log_path, index=False)
    
    def get_next_version(self, algorithm: str) -> str:
        """
        Get next version number for algorithm.
        
        Args:
            algorithm: Algorithm name (e.g., 'lgbm', 'xgb', 'rf')
        
        Returns:
            Version string (e.g., 'v1', 'v2', 'v3')
        """
        # List all model files for this algorithm
        pattern = f"model_{algorithm}_v*.pkl"
        existing_versions = list(self.models_dir.glob(pattern))
        
        if not existing_versions:
            return "v1"
        
        # Extract version numbers and get max
        version_numbers = []
        for path in existing_versions:
            try:
                version_str = path.stem.split('_v')[-1]
                version_numbers.append(int(version_str))
            except (ValueError, IndexError):
                continue
        
        next_version = max(version_numbers) + 1 if version_numbers else 1
        return f"v{next_version}"
    
    def save_model(
        self,
        model: Any,
        algorithm: str,
        metrics: ModelMetrics,
        is_production: bool = False
    ) -> Path:
        """
        Save versioned model and log metrics.
        
        Args:
            model: Trained model object
            algorithm: Algorithm name
            metrics: ModelMetrics dataclass with evaluation scores
            is_production: If True, also save as model_final.pkl
        
        Returns:
            Path to saved model
        """
        # Ensure version is set
        if metrics.version is None:
            metrics.version = self.get_next_version(algorithm)
        
        # Create versioned filename
        model_filename = f"model_{algorithm}_{metrics.version}.pkl"
        model_path = self.models_dir / model_filename
        
        # Save model
        joblib.dump(model, model_path)
        print(f"[OK] Modelo salvo: {model_path}")
        
        # Log metrics
        self._log_metrics(model_filename, metrics)
        
        # If production, create symlink/copy to model_final.pkl
        if is_production:
            final_path = self.models_dir / "model_final.pkl"
            try:
                if final_path.exists():
                    final_path.unlink()
                joblib.dump(model, final_path)
                print(f"[OK] Modelo de produção atualizado: {final_path}")
            except Exception as e:
                print(f"[WARN] Erro ao criar modelo final: {e}")
        
        return model_path
    
    def _log_metrics(self, filename: str, metrics: ModelMetrics):
        """Log metrics to versions_log.csv"""
        # Read existing log
        if self.versions_log_path.exists():
            df_log = pd.read_csv(self.versions_log_path)
        else:
            df_log = pd.DataFrame()
        
        # Create new entry
        entry = {
            'filename': filename,
            'algorithm': metrics.algorithm,
            'version': metrics.version,
            'auc': metrics.auc,
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1': metrics.f1,
            'timestamp': metrics.timestamp,
            'notes': metrics.notes
        }
        
        # Append and save
        df_log = pd.concat([df_log, pd.DataFrame([entry])], ignore_index=True)
        df_log.to_csv(self.versions_log_path, index=False)
        print(f"[OK] Métricas registradas em: {self.versions_log_path}")
    
    def list_models(self) -> pd.DataFrame:
        """List all saved models with metrics."""
        if self.versions_log_path.exists():
            return pd.read_csv(self.versions_log_path)
        return pd.DataFrame()
    
    def load_model(self, filename: str) -> Any:
        """
        Load a specific versioned model.
        
        Args:
            filename: Model filename (e.g., 'model_lgbm_v1.pkl')
        
        Returns:
            Loaded model object
        """
        model_path = self.models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        return joblib.load(model_path)
    
    def load_best_model(self, algorithm: str = None) -> tuple:
        """
        Load the best model by AUC.
        
        Args:
            algorithm: If specified, get best model for that algorithm only
        
        Returns:
            Tuple of (model, metrics_dict)
        """
        if not self.versions_log_path.exists():
            raise FileNotFoundError("Nenhum modelo versionado encontrado")
        
        df_log = pd.read_csv(self.versions_log_path)
        
        if algorithm:
            df_log = df_log[df_log['algorithm'] == algorithm]
        
        if df_log.empty:
            raise ValueError(f"Nenhum modelo encontrado para: {algorithm}")
        
        # Get best by AUC
        best_idx = df_log['auc'].idxmax()
        best_record = df_log.loc[best_idx]
        
        # Load model
        model = self.load_model(best_record['filename'])
        
        return model, best_record.to_dict()


def log_evaluation(
    output_dir: Path = None,
    algorithm: str = "",
    metrics: Dict[str, float] = None,
    report_text: str = ""
):
    """
    Save detailed evaluation report.
    
    Args:
        output_dir: Directory for evaluation reports (default: reports/text)
        algorithm: Algorithm name for filename
        metrics: Dictionary of metrics
        report_text: Full classification report
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "reports" / "text"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_file = output_dir / f"metrics_{algorithm}_{timestamp}.txt"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"Modelo: {algorithm}\n")
        f.write(f"Data: {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        if metrics:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write(report_text)
    
    print(f"[OK] Avaliação salva em: {metrics_file}")
    
    return metrics_file


if __name__ == "__main__":
    # Example usage
    manager = ModelVersionManager()
    
    # List all models
    print("Modelos salvos:")
    print(manager.list_models())
