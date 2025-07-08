# Regression Model Benchmarking

This project provides a comprehensive benchmarking suite for various regression models on the ENB2012 dataset, with a focus on fair and reproducible comparison. All models use MinMaxScaler (feature_range=(0.1, 10.0)) for preprocessing, ensuring consistent evaluation and compatibility with new models that require this scaling.

## Project Structure

```
regression/
├── data/                  # Contains ENB2012_data.csv and related datasets
├── linear_regressor/      # Multi-task Linear Regression implementation
├── sgd_regressor/         # Multi-task SGD Regressor implementation
├── multiout_regressor/    # Multi-output models (Random Forest, Gradient Boosting, etc.)
├── mlp_regressor/         # Multi-layer Perceptron (MLP) regression
├── gp_regressor/          # Multi-task Gaussian Process regression
├── environment.yml        # Conda environment for reproducibility
└── README.md              # This file
```

## Models Implemented

- **Linear Regression**: Multi-task, one model per target
- **SGD Regressor**: Multi-task, one model per target (note: unstable with MinMaxScaler)
- **Multi-output Regressor**: Random Forest, Gradient Boosting, Linear, Ridge, Lasso, SVR
- **MLP Regressor**: Multi-layer Perceptron, with architecture search and hyperparameter tuning
- **Gaussian Process Regressor**: Multi-task, one model per target, with kernel search

## Preprocessing

- **All models use:**
  - `MinMaxScaler(feature_range=(0.1, 10.0))` for feature scaling
  - Pipelines to ensure scaling is applied identically during training, validation, and testing
  - No data leakage: scaling is fit only on training data in each fold or split

## How to Run

1. **Set up the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate regression
   ```

2. **Prepare the data:**
   - Place `ENB2012_data.csv` in the `data/` directory (already present if you cloned the full repo).

3. **Run a model:**
   - Example (Linear Regression):
     ```bash
     python linear_regressor/linear_regressor.py
     ```
   - Replace `linear_regressor` with `sgd_regressor`, `multiout_regressor`, `mlp_regressor`, or `gp_regressor` to run other models.

4. **Outputs:**
   - Results (CSV and plots) are saved in each model's folder, e.g. `linear_regressor/linear_regression_results.csv`.
   - Visualizations and comparison CSVs are also organized per model.

## Results Summary (MinMaxScaler)

- **Best performance:** Gradient Boosting (multi-output) and Gaussian Process
- **MLP and Linear Regression:** Competitive, but not as strong as tree-based or GP models
- **SGD:** Unstable with MinMaxScaler; not recommended without further tuning

## Notes

- All evaluation (test and cross-validation) is performed on MinMax-scaled data.
- To switch to StandardScaler, uncomment the relevant lines in each script.
- For fair comparison with new models, ensure they use the same scaling and evaluation protocol.

## Reproducibility

- All scripts are self-contained and reproducible with the provided environment.
- No data leakage: scaling and model fitting are always performed within proper splits/folds.

## Contact

For questions or contributions, please open an issue or pull request. 