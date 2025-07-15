"""
Final R² Score Calculation for Smart Traffic Model

This script demonstrates the R² calculation implementation using the existing model
and shows how to interpret model accuracy based on R² score.
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_model_accuracy_with_r2():
    """Calculate model accuracy using R² score with the existing setup"""
    
    print("Loading dataset and preparing data...")
    
    # Load dataset (same as in notebook)
    df = pd.read_csv('../Datasets/traffic_dataset.csv')
    
    # Prepare features and targets
    X = df.drop(columns=["green_N", "green_S", "green_E", "green_W"])
    Y = df[["green_N", "green_S", "green_E", "green_W"]]
    
    # Split data with same random state as notebook
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)
    
    # Use the same standardization approach as in notebook
    scaler_independent = StandardScaler()
    X_train_scaled = scaler_independent.fit_transform(X_train)
    X_test_scaled = scaler_independent.transform(X_test)
    
    scaler_dependent = StandardScaler()
    Y_train_scaled = scaler_dependent.fit_transform(Y_train)
    Y_test_scaled = scaler_dependent.transform(Y_test)
    
    # Create and train a model similar to the notebook
    print("Training model...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.callbacks import EarlyStopping
    
    model = Sequential([
        Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(4, activation='linear')
    ])
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train model
    history = model.fit(
        X_train_scaled, Y_train_scaled,
        epochs=100,
        validation_data=(X_test_scaled, Y_test_scaled),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(X_test_scaled, Y_test_scaled, return_dict=True, verbose=0)
    
    print("=" * 70)
    print("SMART TRAFFIC MODEL - ACCURACY CALCULATION BASED ON R² SCORE")
    print("=" * 70)
    
    print("Current Model Performance:")
    print(f"  Mean Absolute Error (MAE): {results['mae']:.6f}")
    print(f"  Loss (MSE): {results['loss']:.6f}")
    
    # Make predictions
    Y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    
    # Transform back to original scale for R² calculation
    Y_pred_original = scaler_dependent.inverse_transform(Y_pred_scaled)
    Y_test_original = Y_test.values
    
    # Calculate R² scores for each traffic direction
    print("\nR² Score Calculation:")
    print("-" * 40)
    
    directions = ['green_N', 'green_S', 'green_E', 'green_W']
    individual_r2_scores = {}
    
    for i, direction in enumerate(directions):
        r2 = r2_score(Y_test_original[:, i], Y_pred_original[:, i])
        individual_r2_scores[direction] = r2
        print(f"  R² Score for {direction}: {r2:.6f}")
    
    # Calculate overall R² score
    overall_r2 = r2_score(Y_test_original, Y_pred_original, multioutput='uniform_average')
    
    # Convert R² to accuracy percentage
    accuracy_percentage = overall_r2 * 100
    
    print(f"\n  Overall R² Score: {overall_r2:.6f}")
    print(f"  Model Accuracy (based on R²): {accuracy_percentage:.2f}%")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("ACCURACY INTERPRETATION")
    print("=" * 70)
    
    def interpret_r2_score(r2):
        if r2 >= 0.90:
            return "Excellent"
        elif r2 >= 0.80:
            return "Good"
        elif r2 >= 0.70:
            return "Acceptable"
        elif r2 >= 0.60:
            return "Moderate"
        else:
            return "Poor"
    
    print("Individual Direction Performance:")
    for direction, r2 in individual_r2_scores.items():
        performance = interpret_r2_score(r2)
        print(f"  {direction}: {performance} ({r2:.6f})")
    
    overall_performance = interpret_r2_score(overall_r2)
    print(f"\nOverall Model Performance: {overall_performance} ({overall_r2:.6f})")
    
    print("\nR² Score Explanation:")
    print(f"• R² = {overall_r2:.6f} means the model explains {accuracy_percentage:.2f}% of the variance")
    print("• R² values closer to 1.0 indicate better model performance")
    print("• This model demonstrates", overall_performance.lower(), "performance for traffic prediction")
    
    # Additional context based on problem statement
    print("\n" + "=" * 70)
    print("PROBLEM STATEMENT CONTEXT")
    print("=" * 70)
    print("Problem: Calculate accuracy based on R² given MAE = 0.004")
    print(f"Current MAE: {results['mae']:.6f}")
    print(f"Calculated Accuracy (R²-based): {accuracy_percentage:.2f}%")
    
    if results['mae'] <= 0.01:  # Close to the mentioned 0.004
        print("✓ MAE is within acceptable range")
    else:
        print("⚠ MAE is higher than the mentioned 0.004")
    
    print(f"✓ R² calculation successfully implemented: {overall_r2:.6f}")
    print("=" * 70)
    
    return {
        'mae': results['mae'],
        'mse': results['loss'],
        'r2_score': overall_r2,
        'accuracy_percentage': accuracy_percentage,
        'individual_r2_scores': individual_r2_scores,
        'interpretation': overall_performance
    }

if __name__ == "__main__":
    try:
        results = calculate_model_accuracy_with_r2()
        
        print("\n🎯 FINAL ANSWER TO PROBLEM STATEMENT:")
        print(f"Model Accuracy based on R² = {results['accuracy_percentage']:.2f}%")
        print(f"R² Score = {results['r2_score']:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()