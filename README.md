# Industrial Equipment Risk Assessment using Ordinal Regression

A machine learning pipeline that combines LSTM time series analysis, Random Forest classification, and ordinal regression to predict industrial equipment risk tiers (Low, Medium, High Risk).

## Overview

This project implements a stacked ensemble approach for classifying factory equipment safety risk levels. The model respects the natural ordering of risk categories using ordinal regression while leveraging both temporal patterns and tabular features.

## Architecture

### Three-Stage Pipeline:

1. **Unsupervised Clustering (PCA + K-Means)**
   - Dimensionality reduction to 3 principal components
   - K-Means clustering (k=3) to identify natural risk groupings
   - Cluster centers ordered by distance from origin to establish risk hierarchy

2. **Feature Extraction Stage**
   - **LSTM Model**: Extracts temporal patterns from sensor time series data
     - Input: Temperature, vibration, pressure, humidity sequences
     - Hidden size: 32 units with 0.7 dropout
     - Output: 32-dimensional feature vectors representing temporal dynamics
   
   - **Random Forest**: Captures tabular feature relationships
     - Constrained depth (max_depth=5) to prevent overfitting
     - Generates probability distributions across risk classes

3. **Ordinal Regression Meta-Learner**
   - LogisticAT model with threshold-based ordinal constraints
   - Input: Concatenated LSTM features + RF probability vectors + ordinal labels
   - Respects inherent ordering: Low Risk < Medium Risk < High Risk

## Key Features

- **Ordinal Constraint Preservation**: Unlike standard classifiers, maintains risk tier ordering
- **Temporal Pattern Recognition**: LSTM captures equipment degradation patterns over time
- **Class Imbalance Handling**: SMOTE oversampling for minority High Risk class
- **Hybrid Feature Space**: Combines deep learning temporal features with ensemble probabilities

## Dataset

- **Source**: Industrial Equipment Monitoring Dataset (Kaggle)
- **Features**: 
  - Numeric: temperature, pressure, vibration, humidity
  - Categorical: equipment type, plant location
- **Target**: Risk levels derived from PCA-based clustering
- **Distribution**: Imbalanced (Low: 60%, Medium: 30%, High: 10%)

## Results

### Model Performance (Test Set)
- **Macro-averaged F1 Score**: 0.8613
- **Overall Accuracy**: 90%

### Per-Class Performance:
| Risk Level | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Low Risk   | 0.93      | 0.89   | 0.91     |
| Medium Risk| 0.89      | 0.90   | 0.89     |
| High Risk  | 0.66      | 0.95   | 0.78     |

**Key Insight**: High recall (95%) on High Risk cases is critical for safety applications, despite lower precision.

## Visualization

The repository includes PCA-based cluster visualizations in both 2D and 3D space, showing clear separation between risk tiers with cluster centroids marked.

## Installation

```bash
pip install torch scikit-learn mord pandas numpy matplotlib imblearn
```

## Usage

```python
# Load and preprocess data
df = pd.read_csv('equipment_anomaly_data.csv')

# Train the pipeline
lstm_model = TorchLSTM(input_size=4, hidden_size=32, num_classes=3)
rf_pipeline = Pipeline([('preprocessor', ...), ('classifier', RandomForestClassifier(...))])

# Extract features and train meta-learner
ordinal_model = LogisticAT(alpha=1.0, max_iter=1000)
ordinal_model.fit(stacked_features_train, y_train, sample_weight=sample_weights)

# Predict risk levels
y_pred = ordinal_model.predict(stacked_features_test)
```

## Technical Considerations

- **Data Leakage Prevention**: SMOTE applied only after train/test split
- **Regularization**: Dropout in LSTM (0.7), constrained RF depth, L1 penalty in ordinal regression
- **Sample Weighting**: Balanced class weights compensate for distribution skew

## Future Improvements

- Implement LSTM sequence padding for variable-length time series
- Experiment with ordinal-aware loss functions in LSTM training
- Add feature importance analysis for interpretability
- Explore semi-supervised learning given limited High Risk examples

## Dependencies

- PyTorch 2.x
- scikit-learn
- mord (ordinal regression)
- imbalanced-learn
- pandas, numpy, matplotlib

## License

MIT

---

**Note**: This model prioritizes safety by maximizing High Risk detection (95% recall) at the cost of some false positives, appropriate for industrial safety applications where missed critical failures are costlier than false alarms.
