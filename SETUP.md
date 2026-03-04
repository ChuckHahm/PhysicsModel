# ML Tabular Pipeline — Environment Setup

## CPU-Only

```bash
conda create -n ml_tabular python=3.11 numpy pandas scikit-learn pytorch torchvision torchaudio joblib -c pytorch -y
```

## CUDA GPU

```bash
conda create -n ml_tabular python=3.11 numpy pandas scikit-learn pytorch torchvision torchaudio pytorch-cuda=12.1 joblib -c pytorch -c nvidia -y
```

## Activate & Run

```bash
conda activate ml_tabular
python ml_tabular_pipeline.py
```

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | `ColumnTransformer`, `StandardScaler`, `OrdinalEncoder`, train/test split |
| `pytorch` | `Dataset`, `DataLoader`, model, training loop |
| `joblib` | Saving/loading the fitted preprocessor (ships with scikit-learn) |
