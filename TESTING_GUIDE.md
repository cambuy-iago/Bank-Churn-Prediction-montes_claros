# 🧪 Guia de Testes - Melhorias Implementadas

## Teste 1: Verificar Streamlit App Melhorado ✨

### Pré-requisitos
- Modelo treinado em `models/model_final.pkl`
- Dependências instaladas: `streamlit`, `pandas`, `joblib`

### Executar
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run webapp/app.py
```

### O que Verificar
✅ Layout em 3 colunas (Demográfico, Atividade, Risco)
✅ Expandível "Parâmetros Avançados"
✅ Métrica do modelo exibida (AUC: 0.9826)
✅ Barra de progresso colorida da probabilidade
✅ Classificação de risco em 4 níveis (cores diferentes)
✅ Gráfico de feature importance com top 5
✅ Recomendações dinâmicas baseadas no perfil
✅ Botões de download JSON e CSV funcionando
✅ Sem erros de carregamento do modelo

---

## Teste 2: Sistema de Versionamento 📦

### Executar Script de Treinamento
```powershell
python src/train_lgbm_enhanced.py
```

### O que Verificar
✅ Exibe progresso [1/5] a [5/5]
✅ Cria `models/model_lgbm_v1.pkl`
✅ Cria `models/model_final.pkl` (é_produção)
✅ Cria/atualiza `models/versions_log.csv`
✅ Salva figuras em `reports/figures/`
✅ Salva relatório em `reports/text/`
✅ Exibe resumo final com paths

### Verificar Arquivo de Log
```powershell
cat models/versions_log.csv
```
**Esperado:**
```csv
filename,algorithm,version,auc,accuracy,precision,recall,f1,timestamp,notes
model_lgbm_v1.pkl,lgbm,v1,0.9826,0.965,0.912,0.883,0.897,2025-12-08T...,12-feature baseline...
```

---

## Teste 3: Notebook de Best Practices 📚

### Abrir Notebook
```powershell
jupyter notebook notebooks/Model_Training_Best_Practices.ipynb
```

### Verificar Seções
✅ Célula 1: Imports sem erros
✅ Célula 2-3: Dados carregados e preparados
✅ Célula 4: Features criadas (50+ variáveis)
✅ Célula 5-6: 12-feature subset selecionado
✅ Célula 7-8: Split e CV executados
✅ Célula 9: Modelo treinado
✅ Célula 10: Métricas exibidas (AUC ~0.98)
✅ Célula 11: Gráficos renderizados
✅ Célula 13: Versionamento funcionando
✅ Célula 15: Histórico de versões exibido
✅ Célula 16: Recomendações de negócio apresentadas

---

## Teste 4: Verificar Outputs Organizados 📊

### Estrutura Esperada
```
models/
├── model_final.pkl           ✅ Deve existir
├── model_lgbm_v1.pkl         ✅ Deve existir
└── versions_log.csv          ✅ Deve existir com dados

reports/
├── figures/
│   ├── feature_importance_lgbm_v1.png    ✅
│   ├── confusion_matrix_lgbm_v1.png      ✅
│   ├── roc_curve_lgbm_v1.png             ✅
│   └── feature_importance_reference.png  ✅
└── text/
    └── metrics_lgbm_v1_*.txt             ✅
```

### Verificar Conteúdo
```powershell
# Listar versões
Get-Content models/versions_log.csv

# Verificar tamanho do modelo
(Get-Item models/model_final.pkl).Length / 1MB  # Esperado: ~5-10 MB

# Verificar figuras
Get-ChildItem reports/figures/ -Filter *.png
```

---

## Teste 5: Integração Completa 🔄

### Fluxo End-to-End
1. **Treinar modelo**
   ```powershell
   python src/train_lgbm_enhanced.py
   ```

2. **Verificar versionamento**
   ```powershell
   cat models/versions_log.csv
   ```

3. **Abrir Streamlit**
   ```powershell
   streamlit run webapp/app.py
   ```

4. **Fazer predição**
   - Preencher dados de um cliente
   - Clicar "Prever Evasão"
   - Baixar resultado em JSON

5. **Verificar arquivo salvo**
   ```powershell
   # JSON deve conter: Customer_Age, Probabilidade_Churn, Nivel_Risco, etc.
   Get-Content churn_prediction_*.json -Tail 5
   ```

---

## Teste 6: Verificar Documentação 📖

### Confirmações
✅ `.github/copilot-instructions.md` atualizado
✅ Seção "Class Imbalance Strategy" presente
✅ Seção "Model Versioning & Evaluation Logging" presente
✅ Seção "Streamlit App" atualizada
✅ `IMPLEMENTATION_SUMMARY.md` criado

### Verificar
```powershell
# Buscar por seções-chave
Select-String -Path .github\copilot-instructions.md -Pattern "class_weight"
Select-String -Path .github\copilot-instructions.md -Pattern "ModelVersionManager"
Select-String -Path .github\copilot-instructions.md -Pattern "Real-time predictions"
```

---

## Teste 7: Testes de Regressão ⚙️

### Verificar que Código Antigo Ainda Funciona
```powershell
# Train scripts antigos devem funcionar (se modelo_final.pkl existir)
# python src/train_lgbm.py  # Opcional
# python src/train_xgb.py   # Opcional
# python src/train_rf.py    # Opcional

# Pipeline existente
python src/pipeline_churn.py  # Deve gerar relatórios
```

### Verificar Imports
```python
# Verificar que novos imports funcionam
from src.model_versioning import ModelVersionManager, ModelMetrics
from src.features import criar_variaveis_derivadas

# Verificar que modelo é carregável
import joblib
model = joblib.load("models/model_final.pkl")
print(model.predict([[40, 1, 10000, 100000, 50, 1000, 5000, 0.2, 0.5, 50000, 0, 0]]))
```

---

## Teste 8: Validações de Dados 🔍

### Verificar Consistency
```python
# Todos os arquivos devem ter o mesmo formato de features
import pandas as pd

# Verificar 12-feature subset em todos os arquivos
webapp_features = [
    'Customer_Age', 'Dependent_count', 'Credit_Limit',
    'Total_Trans_Amt', 'Total_Trans_Ct', 'Ticket_Medio',
    'Gasto_Medio_Mensal', 'Rotativo_Ratio', 'Score_Relacionamento',
    'LTV_Proxy', 'Caiu_Valor', 'Caiu_Transacoes'
]

# Verificar em webapp/app.py
grep -r "columns=\[" webapp/app.py  # Deve ter 12 features

# Verificar em train_lgbm_enhanced.py
grep -r "features = \[" src/train_lgbm_enhanced.py  # Deve ter 12 features
```

---

## Troubleshooting 🔧

### Erro 1: "Model not found"
```
❌ FileNotFoundError: models/model_final.pkl not found
✅ Solução: Executar python src/train_lgbm_enhanced.py
```

### Erro 2: "ModuleNotFoundError: model_versioning"
```
❌ ModuleNotFoundError: No module named 'model_versioning'
✅ Solução: Verificar que src/__init__.py existe
✅ Solução: Executar de dentro do projeto root
```

### Erro 3: "versions_log.csv not readable"
```
❌ CSV vazio ou corrompido
✅ Solução: Deletar e criar novo
   rm models/versions_log.csv
   python src/train_lgbm_enhanced.py
```

### Erro 4: Streamlit cache issues
```
❌ Modelo antigo em cache
✅ Solução: Limpar cache do Streamlit
   streamlit cache clear
   streamlit run webapp/app.py
```

---

## Checklist de Validação Final ✅

- [ ] Streamlit app abre sem erros
- [ ] Previsões funcionam e exportam JSON/CSV
- [ ] `models/model_final.pkl` existe
- [ ] `models/versions_log.csv` tem registros
- [ ] `reports/figures/` tem 4+ gráficos PNG
- [ ] Notebook executa todas as 16 células
- [ ] Documentação atualizada e completa
- [ ] Features consistentes entre arquivos
- [ ] Código antigo ainda funciona (regressão)
- [ ] Imports do novo módulo funcionam

---

## Estimativa de Tempo para Testes

| Teste | Tempo |
|-------|-------|
| Teste 1 (Streamlit) | 5 min |
| Teste 2 (Versionamento) | 10 min |
| Teste 3 (Notebook) | 15 min |
| Teste 4 (Outputs) | 5 min |
| Teste 5 (E2E) | 10 min |
| Teste 6 (Documentação) | 5 min |
| Teste 7 (Regressão) | 5 min |
| Teste 8 (Validações) | 5 min |
| **TOTAL** | **~60 minutos** |

---

## Próximas Etapas Recomendadas

1. ✅ **Testes Manuais** (esta seção)
2. ⏳ **Testes Automatizados** (pytest fixtures)
3. ⏳ **CI/CD Integration** (GitHub Actions)
4. ⏳ **Performance Monitoring** (data drift detection)
5. ⏳ **User Acceptance Testing** (stakeholders)

---

**Criado em**: 2025-12-08
**Status**: Pronto para testes
**Próxima revisão**: Após validação completa
