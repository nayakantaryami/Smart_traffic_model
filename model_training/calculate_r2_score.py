"""
R² Score Calculation for Smart Traffic Model

This script calculates the R² (coefficient of determination) score for the trained traffic model
to assess model accuracy. R² represents the proportion of variance in the dependent variable 
that is predictable from the independent variables.

R² = 1 - (SS_res / SS_tot)
where:
- SS_res = Σ(y_true - y_pred)² (residual sum of squares)
- SS_tot = Σ(y_true - y_mean)² (total sum of squares)

Range: R² ∈ [0, 1] (higher values indicate better model performance)
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data_and_model():
    """Load dataset, trained model, and scalers"""
    # Load dataset
    df = pd.read_csv('../Datasets/traffic_dataset.csv')
    
    # Prepare features and targets
    X = df.drop(columns=["green_N", "green_S", "green_E", "green_W"])
    Y = df[["green_N", "green_S", "green_E", "green_W"]]
    
    # Split data (same random_state as in training)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)
    
    # Load scalers
    with open("../models/traffic_models/scaler_independent.pkl", "rb") as file:
        scaler_independent = pickle.load(file)
    
    with open("../models/traffic_models/scaler_dependent.pkl", "rb") as file:
        scaler_dependent = pickle.load(file)
    
    # Scale data
    X_test_scaled = scaler_independent.transform(X_test)
    Y_test_scaled = scaler_dependent.transform(Y_test)
    
    # Load trained model
    model = load_model('../models/traffic_models/traffic_model.h5')
    
    return model, X_test_scaled, Y_test_scaled, Y_test, scaler_dependent

def calculate_r2_scores(model, X_test_scaled, Y_test_scaled, Y_test_original, scaler_dependent):
    """Calculate R² scores for the model"""
    
    # Make predictions
    Y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    
    # Transform predictions back to original scale
    Y_pred_original = scaler_dependent.inverse_transform(Y_pred_scaled)
    
    # Calculate R² for each output (direction)
    directions = ['green_N', 'green_S', 'green_E', 'green_W']
    individual_r2_scores = {}
    
    for i, direction in enumerate(directions):
        r2 = r2_score(Y_test_original.iloc[:, i], Y_pred_original[:, i])
        individual_r2_scores[direction] = r2
        print(f"R² Score for {direction}: {r2:.6f}")
    
    # Calculate overall R² score (average across all outputs)
    overall_r2 = r2_score(Y_test_original, Y_pred_original, multioutput='uniform_average')
    print(f"\nOverall R² Score (Average): {overall_r2:.6f}")
    
    # Calculate MAE for comparison with the notebook
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(Y_test_scaled, Y_pred_scaled)
    print(f"Mean Absolute Error (scaled): {mae:.6f}")
    
    return individual_r2_scores, overall_r2

def interpret_r2_score(r2):
    """Provide interpretation of R² score"""
    if r2 >= 0.90:
        return "Excellent model performance"
    elif r2 >= 0.80:
        return "Good model performance"
    elif r2 >= 0.70:
        return "Acceptable model performance"
    elif r2 >= 0.60:
        return "Moderate model performance"
    else:
        return "Poor model performance - consider model improvement"

def main():
    """Main function to calculate and display R² scores"""
    print("=" * 60)
    print("Smart Traffic Model - R² Score Calculation")
    print("=" * 60)
    
    try:
        # Load data and model
        print("Loading data and model...")
        model, X_test_scaled, Y_test_scaled, Y_test_original, scaler_dependent = load_data_and_model()
        
        # Calculate R² scores
        print("\nCalculating R² scores...")
        individual_r2, overall_r2 = calculate_r2_scores(
            model, X_test_scaled, Y_test_scaled, Y_test_original, scaler_dependent
        )
        
        # Provide interpretation
        print("\n" + "=" * 60)
        print("Model Performance Interpretation:")
        print("=" * 60)
        
        for direction, r2 in individual_r2.items():
            interpretation = interpret_r2_score(r2)
            print(f"{direction}: {interpretation} (R² = {r2:.6f})")
        
        print(f"\nOverall Model: {interpret_r2_score(overall_r2)} (R² = {overall_r2:.6f})")
        
        print("\n" + "=" * 60)
        print("Note: R² values closer to 1.0 indicate better model performance.")
        print("R² represents the proportion of variance in the target variables")
        print("that is predictable from the input features.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model and data files exist in the correct locations.")

if __name__ == "__main__":
    main()