# logreg-toy dataset (v1)

Synthetic 2D binary classification dataset for logistic regression smoke tests.

- **Positive blob**: mean (1.5, 1.5), diagonal covariance (0.5, 0.5).
- **Negative blob**: mean (-1.0, -1.0), diagonal covariance (0.75, 0.75).
- **Splits**: train=800 samples, test=200 samples.
- **Seed**: 20251005 (JAX PRNG).

Files:
- `train.parquet`
- `test.parquet`
- `metadata.json`

Generation script: `scripts/generate_logreg_toy_dataset.py`.
