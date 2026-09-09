# Distance Estimator

Regression models that estimate the distance to nearby objects from a single
image, for autonomous-navigation-style perception. The pipeline combines
PCA-reduced pixel features with hand-engineered geometric and texture features,
fed into bagged Random-Forest regressors and selected by out-of-fold cross validation.

**Result: ≈ 9.5 cm out-of-fold mean error.**  

## Approach

**Features**
- PCA-reduced raw image features at multiple scales / downsampling factors.
- Engineered geometric + texture features:
  - horizon-line detection via vertical edge density
  - vanishing-point / left–right symmetry features
  - multi-scale local variance (texture density)

**Model**
- Random-Forest regressors over combined feature sets, across scales,
  neighbourhood size, and preprocessing choices.
- Bagging (multiple forests per configuration) for variance reduction.
- Model selection driven by out-of-fold (OOF) error.

**Diagnostics**
- `analyze_residuals.py` inspects error distributions and residual structure
  to guide feature and model choices.

## Repository structure
```
main.py                # feature extraction, training, model selection
utils.py               # helpers
analyze_residuals.py   # residual / error analysis
config.yaml            # configuration
requirements.txt
data/                  # images + labels
```

## Running
```bash
pip install -r requirements.txt
python main.py
```