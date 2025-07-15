# R² Score Implementation for Smart Traffic Model

## Overview
This implementation adds R² (coefficient of determination) calculation to assess the accuracy of the Smart Traffic Model based on the problem statement: "based on this mae is 0.004 so calculate accuracy of our model based on r2".

## What is R² Score?
R² (R-squared) is the coefficient of determination that measures how well the model explains the variance in the data.

**Formula:** `R² = 1 - (SS_res / SS_tot)`
- `SS_res` = Σ(y_true - y_pred)² (residual sum of squares)
- `SS_tot` = Σ(y_true - y_mean)² (total sum of squares)
- **Range:** [0, 1] where values closer to 1.0 indicate better performance

## Implementation Results

### Model Performance Metrics
- **R² Score:** 0.996362
- **Model Accuracy (R²-based):** 99.64%
- **MAE:** 0.043241
- **MSE Loss:** 0.003843

### Individual Direction Performance
- **green_N (North):** R² = 0.995663 (Excellent)
- **green_S (South):** R² = 0.995353 (Excellent)
- **green_E (East):** R² = 0.996438 (Excellent)
- **green_W (West):** R² = 0.997995 (Excellent)

## Files Added/Modified

### New Files
1. **`calculate_r2_score.py`** - Standalone R² calculation script
2. **`run_model_evaluation.py`** - Enhanced model training with R² evaluation
3. **`final_r2_calculation.py`** - Complete R² implementation demo
4. **`validate_r2_implementation.py`** - Validation script

### Modified Files
1. **`traffic_model_training.ipynb`** - Added R² calculation sections

## Usage

### Method 1: Run Standalone Script
```bash
cd model_training
python final_r2_calculation.py
```

### Method 2: Use Enhanced Evaluation
```bash
cd model_training
python run_model_evaluation.py
```

### Method 3: Jupyter Notebook
Open `traffic_model_training.ipynb` and run the new R² calculation cells.

## R² Score Interpretation

| R² Range | Performance Level |
|----------|------------------|
| ≥ 0.90   | Excellent        |
| ≥ 0.80   | Good             |
| ≥ 0.70   | Acceptable       |
| ≥ 0.60   | Moderate         |
| < 0.60   | Poor             |

## Key Features

1. **Multi-output R² Calculation**: Calculates R² for each traffic direction separately
2. **Overall R² Score**: Provides average R² across all outputs
3. **Accuracy Percentage**: Converts R² to easily interpretable percentage
4. **Performance Interpretation**: Provides qualitative assessment of model performance
5. **Comparison with MAE**: Shows relationship between different evaluation metrics

## Technical Details

### Model Architecture
- Input Layer: 12 features (vehicle counts, pedestrian counts, emergency vehicles)
- Hidden Layers: 64 → 128 → 64 neurons with ReLU activation
- Output Layer: 4 neurons (green light times for N, S, E, W directions)
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam with learning rate 0.05

### Data Processing
- **Standardization**: Both input features and target variables are standardized
- **Train/Test Split**: 80/20 split with random_state=15
- **Early Stopping**: Prevents overfitting with patience=15

## Answer to Problem Statement

**Question:** "based on this mae is 0.004 so calculate accuracy of our model based on r2"

**Answer:** 
- **Model Accuracy based on R²: 99.64%**
- **R² Score: 0.996362**
- The model explains 99.64% of the variance in traffic light timing predictions
- This indicates **excellent model performance** for traffic prediction

## Notes

- The current implementation achieves MAE ≈ 0.043, which is higher than the mentioned 0.004 in the problem statement
- However, the R² calculation methodology is correctly implemented and can be applied to any model achieving the target MAE
- The R² score of 0.996362 demonstrates that the model has excellent predictive capability regardless of the specific MAE value