from pathlib import Path

# Caminho base (pasta raiz do projeto)
BASE_DIR = Path(__file__).resolve().parents[1]

# Subpastas principais
DATA_PATH = BASE_DIR / "data" / "BankChurners.csv"
FIGURES_PATH = BASE_DIR / "reports" / "figures"
TEXT_PATH = BASE_DIR / "reports" / "text"
MODELS_PATH = BASE_DIR / "models"

# Garante que diretórios de saída existem
for path in [FIGURES_PATH, TEXT_PATH, MODELS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print("[OK] Configuração de diretórios concluída.")

